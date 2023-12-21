
import ctypes
import sys
from time import time
from os import cpu_count, path

from common import GptParams

import llama_cpp
import util


# A LLaMA interactive session
class LLaMAInteract:
	def __init__(self, params: GptParams) -> None:
		# input args
		self.params = params

		if (self.params.seed <= 0):
			self.params.seed = int(time())

		print(f"seed = {self.params.seed}", file=sys.stderr)

		# runtime args
		self.input_consumed = 0
		self.n_past = 0
		self.n_session_consumed = 0
		self.first_antiprompt = []
		self.remaining_tokens = self.params.n_predict
		self.output_echo = self.params.input_echo
		self.multibyte_fix = []

		# model load
		self.lparams = llama_cpp.llama_context_default_params()
		self.lparams.n_ctx = self.params.n_ctx
		self.lparams.n_parts = self.params.n_parts
		self.lparams.seed = self.params.seed
		self.lparams.memory_f16 = self.params.memory_f16
		self.lparams.use_mlock = self.params.use_mlock
		self.lparams.use_mmap = self.params.use_mmap

		self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)
		if (not self.ctx):
			raise RuntimeError(f"error: failed to load model '{self.params.model}'")

		if (self.params.ignore_eos):
			self.params.logit_bias[llama_cpp.llama_token_eos()] = -float("inf")


		# create internal context
		self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

		# Add a space in front of the first character to match OG llama tokenizer behavior
		self.params.prompt = " " + self.params.prompt

		# tokenize the prompt
		self.embd = []
		self.embd_inp = self._tokenize(self.params.prompt)

		if (len(self.embd_inp) > self.n_ctx - 4):
			raise RuntimeError(f"error: prompt is too long ({len(self.embd_inp)} tokens, max {self.params.n_ctx - 4})")


		# number of tokens to keep when resetting context
		if (self.params.n_keep < 0 or self.params.n_keep > len(self.embd_inp) or self.params.instruct):
			self.params.n_keep = len(self.embd_inp)

		# enable interactive mode if reverse prompt or interactive start is specified
		if (len(self.params.antiprompt) != 0 or self.params.interactive_start):
			self.params.interactive = True

		# determine newline token
		self.llama_token_newline = self._tokenize("\n", False)
		self.llama_token_eot = self._tokenize(" [end of text]\n", False)

		# determine antiprompt tokens
		for i in self.params.antiprompt:
			self.first_antiprompt.append(self._tokenize(i, False))

		self.last_n_tokens = [0]*self.n_ctx #TODO: deque doesnt support slices

	# tokenize a prompt
	def _tokenize(self, prompt, bos=True):
		_arr = (llama_cpp.llama_token * ((len(prompt) + 1) * 4))()
		_n = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8", errors="ignore"), _arr, len(_arr), bos)
		return _arr[:_n]

	def set_color(self, c):
		if (self.params.use_color):
			print(c, end="")

	def use_antiprompt(self):
		return len(self.first_antiprompt) > 0

	# generate tokens
	def generate(self):
		while self.remaining_tokens > 0 or self.params.interactive or self.params.n_predict == -1:
			# predict
			if len(self.embd) > 0:
				# infinite text generation via context swapping
				# if we run out of context:
				# - take the n_keep first tokens from the original prompt (via n_past)
				# - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
				if (self.n_past + len(self.embd) > self.n_ctx):
					n_left = self.n_past - self.params.n_keep
					self.n_past = self.params.n_keep

					# insert n_left/2 tokens at the start of embd from last_n_tokens
					_insert = self.last_n_tokens[
						self.n_ctx - int(n_left/2) - len(self.embd):-len(self.embd)
					]
					self.embd = _insert + self.embd
					self.params.path_session = ""

				#print(f'\nemb  >> [{len(self.embd_inp)}], \'{self.embeding_to_str(self.embd_inp)}\'\n')
				# print(f'\neval >> [{len(self.embd)}], \'{self.embeding_to_str(self.embd)}\'\n')
				if (llama_cpp.llama_eval(
					self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.params.n_threads
				) != 0):
					raise Exception("Failed to llama_eval!")


			self.n_past += len(self.embd)
			self.embd = []
			if len(self.embd_inp) <= self.input_consumed: #&& !is_interacting
				# out of user input, sample next token
				top_k = llama_cpp.llama_n_vocab(self.ctx) if self.params.top_k <= 0 else self.params.top_k
				repeat_last_n = self.n_ctx if self.params.repeat_last_n < 0 else self.params.repeat_last_n

				id = 0

				logits = llama_cpp.llama_get_logits(self.ctx)
				n_vocab = llama_cpp.llama_n_vocab(self.ctx)

				# Apply params.logit_bias map
				for key, value in self.params.logit_bias.items():
					logits[key] += value

				_arr = (llama_cpp.llama_token_data * n_vocab)(*[
					llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
					for token_id in range(n_vocab)
				])
				candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

				# Apply penalties
				nl_logit = logits[llama_cpp.llama_token_nl()]
				last_n_repeat = min(len(self.last_n_tokens), repeat_last_n, self.n_ctx)

				_arr = (llama_cpp.llama_token * last_n_repeat)(*self.last_n_tokens[len(self.last_n_tokens) - last_n_repeat:])
				llama_cpp.llama_sample_repetition_penalty(self.ctx, candidates_p,
					_arr,
					last_n_repeat, llama_cpp.c_float(self.params.repeat_penalty))
				llama_cpp.llama_sample_frequency_and_presence_penalties(self.ctx, candidates_p,
					_arr,
					last_n_repeat, llama_cpp.c_float(self.params.frequency_penalty), llama_cpp.c_float(self.params.presence_penalty))

				if not self.params.penalize_nl:
					logits[llama_cpp.llama_token_nl()] = nl_logit

				# Temperature sampling
				llama_cpp.llama_sample_top_k(self.ctx, candidates_p, top_k, min_keep=llama_cpp.c_size_t(1))
				llama_cpp.llama_sample_tail_free(self.ctx, candidates_p, llama_cpp.c_float(self.params.tfs_z), min_keep=llama_cpp.c_size_t(1))
				llama_cpp.llama_sample_typical(self.ctx, candidates_p, llama_cpp.c_float(self.params.typical_p), min_keep=llama_cpp.c_size_t(1))
				llama_cpp.llama_sample_top_p(self.ctx, candidates_p, llama_cpp.c_float(self.params.top_p), min_keep=llama_cpp.c_size_t(1))
				llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
				id = llama_cpp.llama_sample_token(self.ctx, candidates_p)
				# print("`{}`".format(candidates_p.size))

				self.last_n_tokens.pop(0)
				self.last_n_tokens.append(id)

				# replace end of text token with newline token when in interactive mode
				if (id == llama_cpp.llama_token_eos()):
					id = self.llama_token_newline[0]
					self.embd.append(id)
					self.embd_inp += self.first_antiprompt[0]
					self.embd += self.first_antiprompt[0]
				else:
					# add it to the context
					self.embd.append(id)

				# echo this to console
				self.output_echo = True

				# decrement remaining sampling budget
				self.remaining_tokens -= 1
			else:
				# output to console if input echo is on
				self.output_echo = self.params.input_echo

				# some user input remains from prompt or interaction, forward it to processing
				while len(self.embd_inp) > self.input_consumed:
					self.embd.append(self.embd_inp[self.input_consumed])
					self.last_n_tokens.pop(0)
					self.last_n_tokens.append(self.embd_inp[self.input_consumed])
					self.input_consumed += 1
					if len(self.embd) >= self.params.n_batch:
						break

			# display tokens
			if self.output_echo:
				for id in self.embd:
					yield id

			# reset color to default if we there is no pending user input
			if (self.params.input_echo and len(self.embd_inp) == self.input_consumed):
				self.set_color(util.CONSOLE_COLOR_DEFAULT)

			if (self.params.interactive and len(self.embd_inp) <= self.input_consumed):
				# if antiprompt is present, stop
				if (self.use_antiprompt()):
					if True in [
						i == self.last_n_tokens[-len(i):]
						for i in self.first_antiprompt
					]:
						break

				# if we are using instruction mode, and we have processed the initial prompt
				if (self.params.interactive_start):
					break

			# end of text token
			if len(self.embd) > 0 and self.embd[-1] == llama_cpp.llama_token_eos():
				if (not self.params.instruct):
					for i in self.llama_token_eot:
						yield i
					break

			# respect n_predict even if antiprompt is present
			if (self.params.interactive and self.remaining_tokens <= 0 and self.params.n_predict != -1):
				# If we arent in instruction mode, fix the current generation by appending the antiprompt.
				# Makes it so if chat ends prematurely you dont append the AI's text etc.
				if not self.params.instruct:
					self.embd_inp += self.first_antiprompt[0]
				self.n_remain = self.params.n_predict
				break

		self.params.interactive_start = False

	def __enter__(self):
		return self

	def __exit__(self, type, value, tb):
		self.exit()

	def exit(self):
		llama_cpp.llama_free(self.ctx)
		self.set_color(util.CONSOLE_COLOR_DEFAULT)

	# return past text
	def past(self):
		for id in self.last_n_tokens[-self.n_past:]:
			yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf8", errors="ignore")

	def embeding_to_str(self, embd):
		return "".join([llama_cpp.llama_token_to_str(self.ctx, id).decode("utf8", errors="ignore") for id in embd])

	# write input
	def input(self, prompt: str):
		if (self.params.instruct and self.last_n_tokens[-len(self.inp_prefix):] != self.inp_prefix):
			self.embd_inp += self.inp_prefix
		self.embd_inp += self._tokenize(prompt)
		#print('1 enb input >>', self.embeding_to_str(self.embd_inp))
		# print('2 emb >>', self.embeding_to_str(self.embd))


	# write output
	def output(self):
		self.remaining_tokens = self.params.n_predict
		for id in self.generate():
			cur_char = llama_cpp.llama_token_to_str(self.ctx, id)

			# Add remainder of missing bytes
			if None in self.multibyte_fix:
				self.multibyte_fix[self.multibyte_fix.index(None)] = cur_char

			# Return completed utf char
			if len(self.multibyte_fix) > 0 and not None in self.multibyte_fix:
				yield (b"".join(self.multibyte_fix)).decode("utf8")
				self.multibyte_fix = []
				continue

			# Contains multi-byte UTF8
			for num, pattern in [(2, 192), (3, 224), (4, 240)]:
				# Bitwise AND check
				if pattern & int.from_bytes(cur_char, 'little') == pattern:
					self.multibyte_fix = [cur_char] + ([None] * (num-1))

			# Stop incomplete bytes from passing
			if len(self.multibyte_fix) > 0:
				continue

			yield cur_char.decode("utf8")

	# read user input
	def read_input(self):
		out = ""
		while (t := input()).endswith("\\"):
			out += t[:-1] + "\n"
		return out + t + "\n"

	# interactive mode
	def interact(self):
		for i in self.output():
			print(i,end="",flush=True)
		self.params.input_echo = False

		while self.params.interactive:
			self.set_color(util.CONSOLE_COLOR_USER_INPUT)
			if (self.params.instruct):
				print('\n> ', end="")
				self.input(self.read_input())
			else:
				print(self.params.input_prefix, end="")
				self.input(f"{self.params.input_prefix}{self.read_input()}{self.params.input_suffix}")
				print(self.params.input_suffix,end="")
			self.set_color(util.CONSOLE_COLOR_DEFAULT)

			try:
				for i in self.output():
					print(i,end="",flush=True)
			except KeyboardInterrupt:
				self.set_color(util.CONSOLE_COLOR_DEFAULT)
				if not self.params.instruct:
					print(self.params.fix_prefix,end="")
					self.input(self.params.fix_prefix)

	def sequence(self):
		for i in self.output():
			print(i,end="",flush=True)

		for user in [
			f'hello {self.params.input_suffix}\n',
		   'what is the capital of France?\n',
		   'where the kids come from?\n',]:
			self.input(user)
			for i in self.output():
				print(i,end="",flush=True)

if __name__ == "__main__":
	from datetime import datetime

	USER_NAME="USER"
	AI_NAME="ASSISTANT"
	MODEL = "./models/7B/ggml-vicuna-7b-1.1-q5_1.bin"
	N_PREDICTS = 2048
	N_THREAD = 8

	prompt = f"""A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions."""

	params = GptParams(
		n_ctx=2048,
		temp=0.7,
		top_k=40,
		top_p=0.5,
		repeat_last_n=256,
		n_batch=1024,
		repeat_penalty=1.17647,
		model=MODEL,
		n_threads=N_THREAD,
		n_predict=N_PREDICTS,
		use_color=True,
		interactive=True,
		antiprompt=[f"{USER_NAME}:"],
		input_prefix=" ",
		input_suffix=f"{AI_NAME}:",
		prompt=prompt,
		seed=42,
	)

	with LLaMAInteract(params) as m:
		m.sequence()
