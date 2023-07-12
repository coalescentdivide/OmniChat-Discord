import asyncio
import aiohttp
from colorama import Fore, Back, Style
import discord
from discord.ext import commands
from dotenv import load_dotenv
import json
import logging
import openai
import os
import base64
import replicate


logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

load_dotenv()
openai.api_key = os.getenv("API_KEY")
api_key = openai.api_key
openai.api_base = os.getenv("API_URL")
api_base = openai.api_base
ignored_ids = os.getenv("IGNORED_IDS").split(",")
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
allow_commands = os.getenv("ALLOW_COMMANDS").lower() == "true"
admin_id = os.getenv("ADMIN_IDS").split(",")
behavior_name = os.getenv('DEFAULT_BEHAVIOR')
temperature = float(os.getenv("TEMPERATURE"))
frequency_penalty = float(os.getenv("FREQUENCY_PENALTY"))
presence_penalty = float(os.getenv("PRESENCE_PENALTY"))
top_p = float(os.getenv("TOP_P"))



channel_messages = {}
channel_models = {}
responses = {}
command_mode_flag = {}
message_queue_locks = {}
model_list_json = None


def allowed_command(user_id):
    if str(user_id) in admin_id:
        return True
    elif allow_commands:
        return True
    else:
        return False

def list_prompts():
    """Lists prompt filenames in "./prompts" directory with ".txt" extension, removes extension, and returns modified names"""    
    behavior_files = []
    for filename in os.listdir("./prompts"):
        if filename.endswith(".txt"):
            behavior_files.append(os.path.splitext(filename)[0])
    return behavior_files

def load_prompt(filename):
    """Loads a prompt from a file, processes it, and returns JSON data as a Python object."""   
    if not filename.endswith(".txt"):
        filename += ".txt"
    with open(f"./prompts/{filename}", encoding="utf-8") as file:
        lines = file.readlines()
    return json.loads(build_convo(lines))

def load_prompt_claude(filename):
    """Loads a prompt from a file, processes it, and returns a single string."""   
    if not filename.endswith(".txt"):
        filename += ".txt"
    with open(f"./prompts/{filename}", encoding="utf-8") as file:
        prompt = file.readlines()
    return "\n\n".join(prompt)

def save_convo(messages, filename, channel_id):
    """Saves the current memory to a text file"""
    convo_str = de_json(messages)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(convo_str)
        print(f"{Fore.RED}Behavior Saved:\n{Style.DIM}{Fore.GREEN}{Back.WHITE}{convo_str}{Style.RESET_ALL}")
    return f"Behavior saved as `{os.path.splitext(os.path.basename(filename))[0]}`"

def set_model(model_name, model_info, channel_id, message=None):
    max_tokens = int(model_info[model_name] * 0.8)
    channel_models[channel_id] = (model_name, max_tokens)

def de_json(convo):
    """Convert a conversation from JSON to human-readable text."""
    conversation = []
    for message in convo:
        role = message['role']
        content = message['content']
        conversation.append(f"{role}: {content}")
    return "\n".join(conversation)

def build_convo(lines):
    """Converts a conversation in text format to JSON format."""
    conversation = []
    role = "user"
    content = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':', maxsplit=1)
        if len(parts) >= 2:
            if content:
                conversation.append({"role": role, "content": content})
                content = ""
            role = parts[0]
            content = parts[1].strip()
        else:
            content += "\n" + line if content else line
    if content:
        conversation.append({"role": role, "content": content})
    return json.dumps(conversation)


async def forget_mentions(user_channel_key):
    await asyncio.sleep(300)
    if user_channel_key in channel_messages:
        del channel_messages[user_channel_key]
        print(f"{Fore.RED}Forgetting side convo with user {user_channel_key}{Style.RESET_ALL}")


async def get_models():
    global model_list_json
    models_path = 'models.json'
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_base}/models") as response:
            if response.status == 200:
                all_models = await response.json()
                with open(models_path, 'w') as outfile:
                    json.dump(all_models, outfile, indent=4)
                chat_models = [m for m in all_models.get("data", []) if "/v1/chat/completions" in m.get("endpoints", [])]
                chat_models = [m for m in chat_models if m["id"] != "bard"] # bard not yet functional, comment or remove this line when available
                model_info = {}
                for model in chat_models:
                    model_id = model["id"]
                    tokens = model["tokens"]
                    model_info[model_id] = tokens
                model_list_json = list(model_info.keys())
                return model_info
            else:
                return []


async def get_tokens(api_key: str, model: str, messages: list):
    headers = {'Authorization': f"Bearer {api_key}"}
    json_data = {'model': model, 'messages': messages}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(f"{api_base}/chat/tokenizer", json=json_data) as resp:            
            response = await resp.json()
            #print(response)
            return response


async def get_chat_response(model, messages, max_tokens):   
    response = await asyncio.to_thread(
        openai.ChatCompletion.create,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response


async def chat_response(messages, channel_id):
    model, max_tokens = channel_models[channel_id]
    num_tokens_data = await get_tokens(api_key, model, messages)
    num_tokens = num_tokens_data['string']['count']
    remaining_tokens = max_tokens - num_tokens

    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)

    current_behavior = channel_settings.get(str(channel_id), {}).get("behavior", "default")

    if isinstance(current_behavior, list):
        behavior_len = len(current_behavior)
    else:
        behavior_len = 1
    start_index = behavior_len
    while remaining_tokens / max_tokens < 0.2 and len(messages) > behavior_len:
        oldest_tokens_data = await get_tokens(api_key, model, messages[start_index:start_index+1])
        oldest_tokens = oldest_tokens_data['string']['count']
        messages = messages[:start_index] + messages[start_index+1:]
        num_tokens -= oldest_tokens
        remaining_tokens = max_tokens - num_tokens
    else:
        if remaining_tokens / max_tokens >= 0.2:
            response = await get_chat_response(model, messages, max_tokens)
            #print(f"{Style.DIM}{Fore.WHITE}Remaining tokens:{remaining_tokens}{Style.RESET_ALL}")
        #print(f"Current Memory:{messages}")
    return response, messages, remaining_tokens


async def discord_chunker(message, content):
    """Seamless text and code splitter for Discord."""
    async def send_chunk(chunk):
        await message.channel.send(chunk)

    def find_split_index(content, max_length):
        indices = [
            content.rfind('\n\n', 0, max_length),
            content.rfind('\n', 0, max_length),
            content.rfind('. ', 0, max_length),
            content.rfind(' ', 0, max_length)
        ]
        return max(indices)

    max_chunk_length = 1950
    in_code_block = False
    code_block_lang = ""

    if len(content) <= max_chunk_length:
        await message.channel.send(content)
    else:
        chunks = []
        chunk = ""
        for line in content.splitlines(True):
            if line.startswith("```"):
                if not in_code_block:
                    code_block_lang = line[3:].strip()
                else:
                    code_block_lang = ""
                in_code_block = not in_code_block
            if len(chunk) + len(line) > max_chunk_length:
                if in_code_block:
                    split_index = find_split_index(chunk, max_chunk_length - 4)
                    chunks.append(chunk[:split_index].rstrip() + "```")
                    chunk = f"```{code_block_lang}\n" + chunk[split_index:]
                else:
                    split_index = find_split_index(chunk, max_chunk_length)
                    chunks.append(chunk[:split_index].rstrip() + '\u200b' + '\n')
                    chunk = chunk[split_index:].lstrip()
            chunk += line
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            await send_chunk(chunk)


async def handle_image(attachment, image_prompt=None):
    system_prompt = f"You are an AI that takes the output from a text to image model and transforms it into a grammatically correct sentence."
    model = "chat-bison-001"
    messages = []
    if image_prompt is not None:
        image_response = await image_question(image_prompt, attachment)
        prompt = f"Image Question: {image_prompt}\nAnswer: {image_response}\nResponse:"
    else:
        image_response = await image_caption(attachment)
        prompt = f"Image Caption: {image_response}\n'Response:"
    max_retries = 3
    backoff_factor = 2
    for retry_attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=256,
                top_p=top_p,
                temperature=0.3,
            )
            content = response['choices'][0]['message']['content']
            num_tokens_data = await get_tokens(api_key, model, messages)
            num_tokens = num_tokens_data['string']['count']
            print(content)
            return content, num_tokens
        
        except aiohttp.ClientConnectorError:
            if retry_attempt == max_retries - 1:
                raise
            sleep_time = (backoff_factor ** retry_attempt) + 1
            print(f"Rate limited. Retrying in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)


async def image_question(prompt, attachment):
    image_data = await attachment.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"
    output = await asyncio.to_thread(
        replicate.run,
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={"image": image_data_url, "question": prompt},
    )
    print(f"{Style.BRIGHT}{Fore.YELLOW}Image query: {prompt}\nBLIP-2 answer: {output}{Style.RESET_ALL}")
    return output

async def image_caption(attachment):
    image_data = await attachment.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"
    output = await asyncio.to_thread(
        replicate.run,
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={"image": image_data_url, "caption": True},
    )
    print(f"{Style.BRIGHT}{Fore.YELLOW}BLIP-2 Caption: {output}")
    return output


async def get_completion(prompt):
    max_retries = 3
    backoff_factor = 2
    for retry_attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                openai.Completion.create,
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=256,
                top_p=top_p,
                temperature=0.5,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            content = response['choices'][0]['text']
            return content.strip()
        except aiohttp.ClientConnectorError:
            if retry_attempt == max_retries - 1:
                raise
            sleep_time = (backoff_factor ** retry_attempt) + 1
            print(f"Rate limited. Retrying in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)


# Command Functions

async def help(message):
    embed = discord.Embed(title=f"Send a message in this channel to get a response from {message.guild.me.nick}\nReplies to other users or messages that start with ! are ignored", color=0x00ff00)
    embed.add_field(name="load model", value=f"Choose from a list of chat models to converse with", inline=False)
    embed.add_field(name="wipe memory", value=f"Wipes the short-term memory and reloads the current behavior", inline=False)
    embed.add_field(name="new behavior", value=f"Allows the user to set a new behavior to the current memory", inline=False)
    embed.add_field(name="save behavior", value=f"Saves the current memory as a behavior template", inline=False)
    embed.add_field(name="load behavior [behavior name]", value=f"Wipes memory and loads the specified behavior template. If no filename is provided, a list of available behavior templates will be shown. Then respond with the name of the template you wish to load.", inline=False)
    embed.add_field(name="reset", value=f"Wipes memory and loads the default behavior", inline=False)
    embed.add_field(name="@mention", value=f"You can also mention {message.guild.me.nick} outside of this channel. Keep mentioning it to continue that conversation, otherwise a mention convo is erased automatically after 5 minutes", inline=False)
    await message.channel.send(embed=embed)
    return


async def wipe_memory(message):

    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return

    global behavior_name
    user_channel_key = message.channel.id
    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)    
    behavior_name = channel_settings[str(message.channel.id)]["behavior"]
    if 'claude' in channel_settings[str(message.channel.id)]["model"]:
        behavior = [{"role": "user", "content": load_prompt_claude(behavior_name)}]
    else:
        behavior = load_prompt(behavior_name)

    channel_messages[user_channel_key] = behavior
    await message.channel.send(f"Memory wiped! Current Behavior is `{os.path.splitext(os.path.basename(behavior_name))[0]}`")
    print(f"{Fore.RED}Memory Wiped{Style.RESET_ALL}")
    return


async def reset(message):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return

    global behavior_name
    user_channel_key = message.channel.id

    behavior_name = os.getenv("DEFAULT_BEHAVIOR")
    default_model = os.getenv("DEFAULT_MODEL")

    if 'claude' in default_model:
        behavior = [{"role": "user", "content": load_prompt_claude(behavior_name)}]
    else:
        behavior = load_prompt(behavior_name)

    channel_messages[user_channel_key] = behavior

    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)

    channel_settings[str(message.channel.id)]["behavior"] = os.getenv("DEFAULT_BEHAVIOR")
    channel_settings[str(message.channel.id)]["model"] = os.getenv("DEFAULT_MODEL")

    with open('channel_settings.json', 'w') as f:
        json.dump(channel_settings, f, indent=4)

    await message.channel.send(f"Reset to defaults! Current Behavior is `{os.path.splitext(os.path.basename(behavior_name))[0]}` and Model is `{default_model}`")
    print(f"{Fore.RED}Reset to defaults{Style.RESET_ALL}")
    return


async def new_behavior(message, check):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return
    
    command_mode_flag[message.channel.id] = True
    channel_messages[message.channel.id] = []
    embed = discord.Embed(title=f"Write the new behavior", description=(f"Provide a new behavior. Can be a single prompt, or you can provide an example conversation in the following format:\n\nsystem: a system message\nuser: user message 1\nassistant: example response 1\nuser: user message 2\nassistant: example response 2\n\n"), color=0x00ff00)        
    embed.set_footer(text=(f"If you wish to recall your new behavior later, don\"t forget to save it by typing `save behavior`"))
    await message.channel.send(embed=embed)
    msg = await bot.wait_for("message", check=check)
    channel_messages[message.channel.id] = json.loads(build_convo(msg.content.strip().split('\n')))
    async for m in message.channel.history(limit=1):
        if m.author == message.author and m.content:
            last_user_message = m.content
            break

    embed = discord.Embed(title="New behavior Set!", description=last_user_message, color=0x00ff00)
    await message.channel.send(embed=embed)
    return


async def save_behavior(message, check):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return
    
    command_mode_flag[message.channel.id] = True
    await message.channel.send("Name your behavior:")
    msg = await bot.wait_for("message", check=check)
    filename = "prompts/" + msg.content.strip() + ".txt"
    messages = channel_messages[message.channel.id]
    await message.channel.send(save_convo(messages, filename, message.channel.id))


async def load_behavior(message, check=None):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return
    
    behavior_files = list_prompts()
    if not behavior_files:
        await message.channel.send("No behavior files found.")
        return
    
    words = message.content.lower().split()
    if len(words) > 2:
        filename = " ".join(words[2:]).strip().lower()
    elif check is not None:
        command_mode_flag[message.channel.id] = True
        behavior_files_str = "\n".join(behavior_files)
        embed = discord.Embed(title="Which behavior to load?", description=behavior_files_str)
        await message.channel.send(embed=embed)
        msg = await bot.wait_for("message", check=check)
        filename = msg.content.strip().lower()
    else:
        await message.channel.send("No behavior name provided.")
        return
    
    if filename not in [f.lower() for f in behavior_files]:
        await message.channel.send(f"File not found: {filename}")
        return

    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)
    
    if 'claude' in channel_settings[str(message.channel.id)]["model"]:
        behavior_str = load_prompt_claude(filename)
        behavior = [{"role": "user", "content": behavior_str}]
    else:
        behavior = load_prompt(filename)
        convo_str = de_json(behavior)

    channel_messages[message.channel.id] = behavior
    behavior_name = filename

    channel_settings[str(message.channel.id)]["behavior"] = filename
    with open('channel_settings.json', 'w') as f:
        json.dump(channel_settings, f, indent=4)

    embed = discord.Embed(title=f"Behavior loaded: {filename}", description="", color=0x00ff00)
    await message.channel.send(embed=embed)
    print(f"{Fore.RED}Behavior Loaded:\n{Style.DIM}{Fore.GREEN}{Back.WHITE}{convo_str}{Style.RESET_ALL}")




async def load_model(message, check):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return
          
    model_info = await get_models()
    model_ids = list(model_info.keys())
    if not model_ids:
        await message.channel.send("No models found.")
        return
    
    else:
        command_mode_flag[message.channel.id] = True
        model_ids_str = "\n".join(model_ids)
        embed = discord.Embed(title="Which model to load?", description=model_ids_str)
        await message.channel.send(embed=embed)
        msg = await bot.wait_for("message", check=check)
        model_id = msg.content.strip().lower()
        if model_id not in model_ids:
            await message.channel.send(f"Model not found: {model_id}")
            return
        
        else:
            set_model(model_id, model_info, message.channel.id)
            if message.channel.id in channel_messages:
                channel_messages[message.channel.id].clear()

            with open('channel_settings.json', 'r') as f:
                channel_settings = json.load(f)

            current_behavior = channel_settings.get(str(message.channel.id), {}).get("behavior", "default")
            if 'claude' in model_id:
                channel_messages[message.channel.id] = [{"role": "user", "content": load_prompt_claude(current_behavior)}]
            else:
                channel_messages[message.channel.id] = load_prompt(current_behavior)

            channel_settings[str(message.channel.id)]["model"] = model_id

            with open('channel_settings.json', 'w') as f:
                json.dump(channel_settings, f, indent=4)
    print(f"Model {Fore.YELLOW}{Style.BRIGHT}{model_id}{Style.RESET_ALL} loaded in channel {Fore.CYAN}{message.channel.name}{Style.RESET_ALL} in {Fore.GREEN}{message.guild.name}{Style.RESET_ALL}")
    embed = discord.Embed(title=f"Model loaded: {model_id}", description=f"Max tokens: {model_info[model_id]}", color=0x00ff00)
    await message.channel.send(embed=embed)
    return




async def toggle_active(message):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return

    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)
    if str(message.channel.id) not in channel_settings:
        channel_settings[str(message.channel.id)] = {
            "model": "gpt-4",
            "behavior": "default",
            "active": False
        }
    channel_settings[str(message.channel.id)]["active"] = not channel_settings[str(message.channel.id)]["active"]

    with open('channel_settings.json', 'w') as f:
        json.dump(channel_settings, f, indent=4)
    print(f"Channel {Fore.CYAN}{Style.BRIGHT}{message.channel.name}{Style.RESET_ALL} in {Fore.GREEN}{message.guild.name}{Style.RESET_ALL} is now {'active' if channel_settings[str(message.channel.id)]['active'] else 'inactive'}.")
    await message.channel.send(f"This channel is now {'active' if channel_settings[str(message.channel.id)]['active'] else 'inactive'}.")


COMMAND_HANDLERS = {
    "help": help,
    "wipe memory": wipe_memory,
    "new behavior": new_behavior,
    "save behavior": save_behavior,
    "load behavior": load_behavior,
    "load model": load_model,
    "reset": reset
}


@bot.event
async def on_ready():
    print(f"{Fore.GREEN}Logged in as {bot.user}{Style.RESET_ALL}")
    invite_link = discord.utils.oauth_url(bot.user.id, permissions=discord.Permissions(), scopes=("bot", "applications.commands"))
    print(f"{Fore.GREEN}Invite: {Style.BRIGHT}{invite_link}{Style.RESET_ALL}")

    models_path = 'models.json'
    model_info = {}

    if os.path.exists(models_path):
        try:
            with open(models_path, 'r') as f:
                all_models = json.load(f)
                chat_models = [m for m in all_models.get("data", []) if "/v1/chat/completions" in m.get("endpoints", [])]
                chat_models = [m for m in chat_models if m["id"] != "bard"] # bard not yet functional, comment or remove this line when available
                for model in chat_models:
                    model_id = model["id"]
                    tokens = model["tokens"]
                    model_info[model_id] = tokens
                model_list_json = list(model_info.keys())
        except:
            model_info = await get_models()
    else:
        model_info = await get_models()

    with open('channel_settings.json', 'r') as f:
        channel_settings_dict = json.load(f)
    
    for channel_id_str, channel_settings in channel_settings_dict.items():
        channel_id = int(channel_id_str)
        model_name = channel_settings["model"]
        if model_name not in model_info:
            print(f"Default model {model_name} for channel {channel_id} not found.")
        else:
            set_model(model_name, model_info, channel_id)
        filename = channel_settings["behavior"]

        try:
            if 'claude' in channel_settings["model"]:
                channel_messages[channel_id] = [{"role": "user", "content": load_prompt_claude(filename)}]
            else:
                channel_messages[channel_id] = load_prompt(filename)
        except FileNotFoundError:
            print(f"Prompt file {filename} not found for channel {channel_id}.")
        except Exception as e:
            print(f"Error loading prompt file {filename} for channel {channel_id}. Error: {e}")

    print(f"{Fore.BLUE}{Style.BRIGHT}Defaults loaded.{Style.RESET_ALL}")




@bot.event
async def on_message(message):
    global behavior_name

    with open('channel_settings.json', 'r') as f:
        channel_settings = json.load(f)

    await asyncio.sleep(0.1)
    if message.author.bot or message.author.id in ignored_ids:
        return

    if message.content.strip().lower().startswith("toggle active"):
        await toggle_active(message)
        return

    active = channel_settings.get(str(message.channel.id), {}).get("active", False)

    bot_mentioned_in_inactive_channel = not active and bot.user in message.mentions
    other_users_mentioned = any(user.id != bot.user.id for user in message.mentions)

    if not bot_mentioned_in_inactive_channel and other_users_mentioned:
        return

    if not bot_mentioned_in_inactive_channel and not active:
        return
    
    message_content = message.content.replace(f"{bot.user.mention}", '').strip()
    if message.reference is not None or message.content.startswith('!'):
        return

    def check(msg):
        return msg.author == message.author and msg.channel == message.channel

    if bot_mentioned_in_inactive_channel:
        user_channel_key = message.author.id
    else:
        user_channel_key = message.channel.id

    messages = channel_messages.get(user_channel_key)
    if messages is None:
        if 'claude' in channel_settings[str(message.channel.id)]["model"]:
            messages = [{"role": "user", "content": load_prompt_claude(os.getenv("DEFAULT_BEHAVIOR"))}]
        else:
            messages = load_prompt(filename=os.getenv("DEFAULT_BEHAVIOR"))
        channel_messages[user_channel_key] = messages

    attachment = message.attachments[0] if message.attachments else None

    words = message_content.lower().split()
    for idx in range(len(words)):
        command = " ".join(words[:idx + 1])
        if command in COMMAND_HANDLERS:
            handler = COMMAND_HANDLERS[command]
            if handler.__code__.co_argcount > 1:
                await handler(message, check)
            else:
                await handler(message)
            return
        
    else:
        if command_mode_flag.get(message.channel.id):
            command_mode_flag[message.channel.id] = False
            return
        
        if user_channel_key not in message_queue_locks:
            message_queue_locks[user_channel_key] = asyncio.Lock()
        image_prompt = message_content.strip() if attachment and message_content.strip() else None
       # messages.append({"role": "user", "content": message_content})

    async with message_queue_locks[user_channel_key]:
        async with message.channel.typing():
            _, max_tokens = channel_models[message.channel.id]  # Retrieve max_tokens
            remaining_tokens = max_tokens  # Initialize remaining_tokens
            if 'claude' in channel_settings[str(message.channel.id)]["model"]:
                messages.append({"role": "user", "content": "name, " + message.author.nick.capitalize() + ": " + message_content + "\n\nAssistant: "})
            else:
                messages.append({"role": "user", "content": "name, " + message.author.nick.capitalize() + ": " + message_content})
            max_retries = 3
            for i in range(max_retries):
                try:
                    if attachment:
                        content, num_tokens = await handle_image(attachment, image_prompt)
                        remaining_tokens -= num_tokens
                    else:
                        response, messages, remaining_tokens = await chat_response(messages, message.channel.id)
                        if response is None:
                            await message.channel.send("No model loaded for this channel. Please load a model using the `load model` command.")
                            return
                        content = response['choices'][0]['message']['content']
                    if 'claude' in channel_settings[str(message.channel.id)]["model"]:
                        messages.append({"role": "assistant", "content": content + "\n\nHuman: "})
                    else:
                        messages.append({"role": "assistant", "content": content})

                    print(f"{Style.DIM}{Fore.RED}Remaining tokens:{remaining_tokens}{Style.RESET_ALL}\nCurrent Memory:{messages}")
                    print(f"Channel: {message.channel.name}\n{Style.DIM}{Fore.RED}{Back.WHITE}{message.author}: {Fore.BLACK}{message_content}{Style.RESET_ALL}\n{Style.DIM}{Fore.GREEN}{Back.WHITE}{bot.user}: {Fore.BLACK}{content}{Style.RESET_ALL}")
                    
                    responses[message.id] = content
                    if message.id in responses:
                        response_content = responses[message.id]
                        await discord_chunker(message, response_content)
                    if bot_mentioned_in_inactive_channel:
                        asyncio.create_task(forget_mentions(user_channel_key))
                    break

                except openai.error.APIError as e:
                    print(f"Retry {i+1}:", type(e), e)
                    if i < max_retries - 1:
                        await asyncio.sleep(2 ** i)
                    else:
                        await message.channel.send("Say again?.")

                except aiohttp.ContentTypeError as e:
                    print(f"Retry {i+1}:", type(e), e)
                    if i < max_retries - 1:
                        await asyncio.sleep(2 ** i)
                    else:
                        await message.channel.send("Sorry, what was that?")
                except aiohttp.ClientConnectorError as e:
                    print(f"Retry {i+1}:", type(e), e)
                    if i == max_retries - 1:
                        await message.channel.send("Huh?")
                except Exception as e:
                    print(type(e), e)

            else:
                logging.error(f"{message.author.name}|{message.channel.name}", exc_info=True)
                await message.channel.send("Sorry, there was an error processing your message.")



bot.run(os.getenv("DISCORD_TOKEN"))
