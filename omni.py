import asyncio
import base64
import io
import json
import logging
import os

import aiohttp
import discord
import openai
import tiktoken
from colorama import Back, Fore, Style
from discord.ext import commands
from dotenv import load_dotenv
from imaginepy import AsyncImagine
from PyPDF2 import PdfReader

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
imagine = AsyncImagine()


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


def num_tokens_from_message(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == model:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


async def get_tokens_api(api_key: str, model: str, messages: list):
    headers = {'Authorization': f"Bearer {api_key}"}
    json_data = {'model': model, 'messages': messages}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(f"{api_base}/chat/tokenizer", json=json_data) as resp:            
            response = await resp.json()
            #print(response)
            return response
        

async def forget_mentions(user_channel_key): # todo: improve this
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
                chat_models = [m for m in all_models.get("data", []) if "/api/v1/chat/completions" in m.get("endpoints", [])]
                model_info = {}
                for model in chat_models:
                    model_id = model["id"]
                    if 'tokens' in model:
                        tokens = model["tokens"]
                        model_info[model_id] = tokens
                model_list_json = list(model_info.keys())
                return model_info
            else:
                return []


async def get_chat_response(model, messages, max_tokens):   
    response = await asyncio.to_thread(
        openai.ChatCompletion.create,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response


async def chat_response(messages, channel_id): #todo: setup local tokenizer for llama
    model, max_tokens = channel_models[channel_id]
    #num_tokens_data = await get_tokens_api(api_key, model, messages)
    #print(num_tokens_data)
    #num_tokens = num_tokens_data['string']['count']
    num_tokens = num_tokens_from_message(messages)
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
        #oldest_tokens_data = await get_tokens_api(api_key, model, messages[start_index:start_index+1])
        #oldest_tokens = oldest_tokens_data['string']['count']
        oldest_tokens = num_tokens_from_message(messages[start_index:start_index+1])
        messages = messages[:start_index] + messages[start_index+1:]
        num_tokens -= oldest_tokens
        remaining_tokens = max_tokens - num_tokens
    else:
        if remaining_tokens / max_tokens >= 0.2:
            response = await get_chat_response(model, messages, max_tokens)

    return response, messages, remaining_tokens


async def discord_chunker(message, content):
    """(Almost) Seamless text and code splitter for Discord."""
    
    async def send_chunk(chunk):
        await message.channel.send(chunk)

    def find_split_index(content, max_length, in_code_block):
        if in_code_block:
            indices = [
            content.rfind('\n', 0, max_length)  # Only consider the lines in the code blocks
            ]
        else:
            indices = [
                content.rfind('\u200b', 0, max_length),
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
            if line.startswith('```'):
                if not in_code_block:
                    code_block_lang = line[3:].strip()
                else:
                    code_block_lang = ""
                in_code_block = not in_code_block

            if len(chunk) + len(line) > max_chunk_length:
                if in_code_block:
                    split_index = find_split_index(chunk, max_chunk_length - 4, in_code_block)
                    chunks.append(chunk[:split_index].rstrip() + '```')
                    chunk = f'```{code_block_lang}\n' + chunk[split_index:]
                else:
                    split_index = find_split_index(chunk, max_chunk_length, in_code_block)
                    chunks.append(chunk[:split_index].rstrip())
                    chunk = '\u200b\n' + chunk[split_index:].lstrip()
            chunk += line
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            await send_chunk(chunk)


async def handle_image(attachment, image_prompt=None):
    if image_prompt is not None:
        image_response = await image_question(image_prompt, attachment)
    else:
        image_response = await image_caption(attachment)    
    return image_response


async def image_question(prompt, attachment):
    image_data = await attachment.read()
    caption = await imagine.interrogator(image_data)
    print(f"{Fore.YELLOW}Interrogator caption: {caption}{Style.RESET_ALL}")
    prompt_text = f"A user is asking a question about an image. This is their question: {prompt}\nThis is the raw output caption from an image to text model:\n`{caption}`\n Using this data returned from that model, attempt to answer the user's question in a complete sentence. Don't assume who made the image, and only mention 'in the style of' if artists are mentioned. The caption will be raw and contain multiple answers. The first block of information is the most accurate description:"
    response = await get_completion(prompt_text)
    await imagine.close()
    return response


async def image_caption(attachment):
    image_data = await attachment.read()
    caption = await imagine.interrogator(image_data)
    print(f"{Fore.YELLOW}Interrogator caption: {caption}{Style.RESET_ALL}")
    prompt_text = f"The following text is the raw output from an image to text model. Using this text, attempt to describe the image it represents in a complete sentence. Don't assume who made the image, and only mention 'in the style of' if artists are mentioned. The first block of information is the most accurate description:\n`{caption}`"
    response = await get_completion(prompt_text)
    await imagine.close()
    return response


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
                temperature=0.5,
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
    bot_name = message.guild.me.nick or message.guild.me.name

    title = f"I'm {bot_name} the chat bot!"
    footer = f"Made with ❤️ by Trypsky "

    embed = discord.Embed(
        title=f"{title}", color=0x00ff00
    )
    embed.set_footer(text=footer)

    embed.add_field(name="🗣️ Chat", value=" Send messages in an active channel to have a chat! Unless you reply to or @mention another user, then I'll keep quiet. 🤫 I also ignore messages that start with `!`", inline=False)
    embed.add_field(name="📣 Quick Mention", value=f"I'm always here for a quick question. Just call me out with a @mention in a non active channel! I will forget about what we talked about in a few minutes though.\n\u200b", inline=False)

    if allowed_command(message.author.id):
        # Additional fields for admin users
        embed.add_field(name="Commands (admin only)", value=f"Below are the available commands. To use them, just type the words, no need for symbols or slashes:", inline=False)
        embed.add_field(name="toggle active 🔄", value=f"Set a channel as active or inactive.", inline=False)
        embed.add_field(name="load model 🧠", value=f"Tired of my current conversation style? Choose from a list of chat models! Changing the model wipes my memory!", inline=False)
        embed.add_field(name="wipe memory 🧹", value=f"Clear my short-term memory.", inline=False)
        embed.add_field(name="new behavior 🎭", value=f"Write a new behaviour.", inline=False)
        embed.add_field(name="save behavior 💾", value=f"After sending your new behavior, type save behavior, followed by a name. You can then recall that behavior later!", inline=False)
        embed.add_field(name="load behavior 🔄", value=f"Type load behavior to see the full list of available behaviors. Then just respond with the name of the one you want! If you know the behavior name already you can type it all in one line like `load behavior default`.", inline=False)
        embed.add_field(name="reset 🔘", value=f"Wipe my memory and load the default behavior.", inline=False)

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
    embed.set_footer(text=(f"If you wish to recall your new behavior later, don't forget to save it by typing `save behavior`"))
    await message.channel.send(embed=embed)
    msg = await bot.wait_for("message", check=check)

    text_attachment_used = False

    if msg.attachments:
        if msg.attachments[0].filename.endswith('.txt'):
            try:
                behavior_text = await msg.attachments[0].read()
                behavior_text = behavior_text.decode('utf-8')
                text_attachment_used = True
            except Exception as e:
                embed = discord.Embed(title="Failed to read the attachment!", description=str(e), color=0xff0000)
                await message.channel.send(embed=embed)
                return
        else:
            embed = discord.Embed(title="Unsupported file type!", description="Please attach a .txt file.", color=0xff0000)
            await message.channel.send(embed=embed)
            return
    else:
        behavior_text = msg.content.strip()

    channel_messages[message.channel.id] = json.loads(build_convo(behavior_text.split('\n')))

    if not text_attachment_used:
        async for m in message.channel.history(limit=1):
            if m.author == message.author and m.content:
                last_user_message = m.content
                break
        embed_description = last_user_message
    else:
        embed_description = "New behavior Set! Type `save behavior` if you want to keep it!"

    embed = discord.Embed(title="New behavior Set! Type `save behavior` if you want to keep it!", description=embed_description, color=0x00ff00)
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
    

    behavior = load_prompt(filename)
    behavior_str = de_json(behavior)

    channel_messages[message.channel.id] = behavior
    behavior_name = filename

    channel_settings[str(message.channel.id)]["behavior"] = filename
    with open('channel_settings.json', 'w') as f:
        json.dump(channel_settings, f, indent=4)

    embed = discord.Embed(title=f"Behavior loaded: {filename}", description="", color=0x00ff00)
    await message.channel.send(embed=embed)
    print(f"{Fore.RED}Behavior Loaded:\n{Style.DIM}{Fore.GREEN}{Back.WHITE}{behavior_str}{Style.RESET_ALL}")


async def load_model(message, check):
    if not allowed_command(message.author.id):
        await message.channel.send("You are not allowed to use this command.")
        return
          
    try:
        model_info = await get_models()
        model_ids = list(model_info.keys())
    except AttributeError:
        await message.channel.send("Sorry, api is currently down, try again later!")
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
            "model": "gpt-3.5-turbo",
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

    if not os.path.exists('channel_settings.json'):
        with open('channel_settings.json', 'w') as json_file:
            json.dump({}, json_file)

    models_path = 'models.json'
    model_info = {}

    try:
        with open(models_path, 'r') as f:
            all_models = json.load(f)
            chat_models = [m for m in all_models.get("data", []) if "/api/v1/chat/completions" in m.get("endpoints", [])]
            for model in chat_models:
                model_id = model["id"]
                tokens = model["tokens"]
                model_info[model_id] = tokens
            model_list_json = list(model_info.keys())
    except:
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

    if (message.reference and not message.reference.resolved.attachments) or message.content.startswith('!'):
        return

    def check(msg):
        return msg.author == message.author and msg.channel == message.channel

    if bot_mentioned_in_inactive_channel:
        user_channel_key = message.author.id
    else:
        user_channel_key = message.channel.id

    messages = channel_messages.get(user_channel_key)
    if messages is None:

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


    async with message_queue_locks[user_channel_key]:
        async with message.channel.typing():

            _, max_tokens = channel_models[message.channel.id]
            remaining_tokens = max_tokens

            if message.author.nick is not None:
                nickname = message.author.nick.split('#')[0].capitalize()
            else:
                nickname = message.author.name.split('#')[0].capitalize()

            messages.append({"role": "user", "content": nickname + ": " + message_content})

            if message.reference:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                attachment = original_message.attachments[0] if original_message.attachments else None

            max_retries = 3
            for i in range(max_retries):
                try:

                    if attachment:
                        image_extensions = ('png', 'jpg', 'jpeg', 'gif')
                        file_extension = attachment.filename.lower().split('.')[-1]
                        if file_extension in image_extensions:
                            content = await handle_image(attachment, image_prompt)

                        else:
                            # Handle text and PDF file attachment
                            file_content = await attachment.read()
                            if file_extension == 'pdf':
                                file_io = io.BytesIO(file_content)
                                pdf = PdfReader(file_io)
                                extracted_text = []
                                for page_num in range(len(pdf.pages)):
                                    page = pdf.pages[page_num]
                                    page_text = page.extract_text()
                                    page_text_with_number = f'Page {page_num + 1}:\n{page_text}'
                                    extracted_text.append(page_text_with_number)
                                file_content = "\n".join(extracted_text)
                                file_content = file_content.encode('utf-8')
                            message_content_for_file = "Document: " + file_content.decode('utf-8')
                            query_document = [{"role": "user", "content": message_content + "\n\n" + message_content_for_file}]

                            # Get the token count of the file content
                            model = channel_settings[str(message.channel.id)]["model"]
                            #num_tokens_data = await get_tokens_api(api_key, model, query_document)
                            #num_tokens_doc = num_tokens_data['string']['count']
                            num_tokens_doc = num_tokens_from_message(query_document)
                            print(f"{Fore.YELLOW}Document loaded with token count of:{num_tokens_doc}{Style.RESET_ALL}")

                            # Check if the content exceeds the token limit
                            max_content_length = remaining_tokens - 500
                            if num_tokens_doc > max_content_length:
                                print(f"{Fore.YELLOW}Document too large for current model!{Style.RESET_ALL}") # todo: use vectorstore                                
                                await message.channel.send("Document too large for current model!")
                                return
                               
                            else:
                                messages.append({"role": "user", "content": message_content + "\n\n" + message_content_for_file})
                                response, messages, remaining_tokens = await chat_response(messages, message.channel.id)
                                content = response['choices'][0]['message']['content']

                    else:
                        response, messages, remaining_tokens = await chat_response(messages, message.channel.id)
                        if response is None:
                            await message.channel.send("No model loaded for this channel. Please load a model using the `load model` command.")
                            return
                        content = response['choices'][0]['message']['content']

                    messages.append({"role": "assistant", "content": content})

                    print(f"{Style.DIM}{Fore.RED}Remaining tokens:{remaining_tokens}{Style.RESET_ALL}{Style.DIM}\nCurrent Memory:{messages}{Style.RESET_ALL}")
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

                except aiohttp.ContentTypeError as e:
                    print(f"Retry {i+1}:", type(e), e)

                except aiohttp.ClientConnectorError as e:
                    print(f"Retry {i+1}:", type(e), e)

                except Exception as e:
                    print(type(e), e)

            else:
                logging.error(f"{message.author.name}|{message.channel.name}", exc_info=True)
                await message.channel.send("I'm unable to respond right now, I'll be back!")


bot.run(os.getenv("DISCORD_TOKEN"))
