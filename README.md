# OmniChat-Discord
Discord Chatbot featuring multiple LLMs, configurable behaviors, self adjusting memory and image understanding. Set up multiple channels with different behaviors and/or chat models!

Built for [Chimeragpt](https://chimeragpt.adventblocks.cc/)

## Prerequisites

- discord bot

- python environment

- [chimeragpt](https://chimeragpt.adventblocks.cc/) api key

# Installation and usage

- Setup a discord bot with all intents. make the bot private

- Create and activate a python venv, then install requirements.txt

- Rename .env.example to .env and add your discord token, chimera api key, and any user id's that should be allowed to use the commands. Change `ALLOW_COMMANDS=false` to true if you want all users to be able to use the commands

- From the activated venv run `python omni.py`. An link will be shown to invite the bot to your server

- Pick a channel or create a new one, and type `toggle active` to activate that channel for chatting! Once active, messages will be passed to the chatbot. The bot will ignore messages that mention other users (or replies to other users) It will also ignore messages that start with !, which is useful if you have other bots that use ! as the command trigger

- Type `help` to see all the available commands

- Type `load model` in an active channel to see the available models, and then type back the model you want to load.

- Behaviors are stored in the \prompts folder as txt files. You can write your own directly and save them there, or you can use the commands to create and save behaviors from within discord. 

- Type `load behavior` to load a saved behavior. This along with the loaded model will be stored in channel_settings.json to preserve the behavior and model between bot restarts. 
