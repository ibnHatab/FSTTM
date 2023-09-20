
import dataclasses
from typing import List, Any, Dict


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep: str
    sep2: str = None

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep=self.sep,
            sep2=self.sep2,
        )

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.
        """
        self.messages[-1][1] = message

# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

register_conv_template(
    Conversation(
            name="vicuna_v1.1",
            system="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            messages=(),
            offset=0,
            sep=" ",
            sep2="</s>",
        )
)
register_conv_template(
    Conversation(
            name="llama_2_13b-chat",
            system="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            roles=("USER", "ASSISTANT"),
            messages=(),
            offset=0,
            sep=" ",
            sep2="</s>",
        )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
