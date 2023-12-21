#!/bin/bash

# Define a list of short strings
sentences=("Hello there!" "How are you?" "This is a Bash script" "Looping through sentences")

# Iterate over the list and echo each sentence
for sentence in "${sentences[@]}"; do
    echo "$sentence" | tee `strace -o spork tty | tr -d '\n\r'` | RHVoice-client  -s  SLT -r 0.6 -v -0.1 | aplay
    sleep 0.1
done

#