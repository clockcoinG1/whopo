import sqlite3

import matplotlib.pyplot as plt

# connect to the database
get_db_path = "/Users/clockcoin/Library/Messages/chat.db"
conn = sqlite3.connect("/Users/clockcoin/Library/Messages/chat.db")
c = conn.cursor()

# retrieve data
# c.execute(
#     "SELECT id, COUNT(*) FROM message JOIN handle ON message.handle_id = handle.ROWID WHERE handle.service = 'iMessage' AND message.is_from_me = 0 GROUP BY message.handle_id"
# )
c.execute(
    "SELECT COUNT(*), strftime('%H', date/1000000000 + 978307200, 'unixepoch') FROM message WHERE date IS NOT NULL GROUP BY strftime('%H', date/1000000000 + 978307200, 'unixepoch')"
)
data = c.fetchall()
x = [row[1] for row in data]
y = [len(row[0]) for row in data]
plt.scatter(x, y)
plt.xlabel("Time of day")
plt.ylabel("Message length")
plt.title("Relationship between message length and time of day")
plt.show()
# create a bar chart
labels = [row[0] for row in data]
sizes = [row[1] for row in data]
plt.bar(labels, sizes)
plt.xlabel("User")
plt.ylabel("Number of messages")
plt.title("Number of messages sent to each user")
plt.show()
c.execute("select text, date/1000000000 FROM message WHERE text IS NOT NULL")
data = c.fetchall()

# close the database
# conn.close()

# create a scatterplot
x = [row[1] for row in data]
y = [len(row[0]) for row in data]
plt.scatter(x, y)
plt.xlabel("Time of day")
plt.ylabel("Message length")
plt.title("Relationship between message length and time of day")
plt.show()

from subprocess import PIPE, Popen


# function to send iMessage
def send_imessage(
    phone_number,
    message,
):
    p = Popen(["ascript", "-"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    script = f'tell application "Messages" \n send "{message}" to buddy "{phone_number}" \n end tell\n'
    stdout, stderr = p.communicate(script.encode())
    return stdout.decode(), stderr.decode()


# function to read iMessage
def read_imessage():
    p = Popen(["osascript", "-"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    script = """
    set imessages to {}
    tell application "Messages"
        repeat with i from 1 to count of imessages
            set this_message to item i of imessages
            set message_text to content of this_message
            set message_sender to sender of this_message
            if message_sender is not "You" then
                set phone_number to phone number of message_sender
                return phone_number & ":" & message_text
            end if
        end repeat
    end tell
    """.format(
        'every message of (chat 1 whose name = "Appleseed, Johnny")'
    )
    stdout, stderr = p.communicate(script.encode())
    return stdout.decode(), stderr.decode()


# main function to run the script
def main():
    while True:
        message = input("Enter your message: ")
        if message == "exit":
            break
        send_imessage("+1-234-567-8901", message)
        response = read_imessage()
        print("Received response: {}".format(response))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import re
import subprocess
import time

# Define the name of the application to open
app_name = "Messages"

# Define the contact name to send a message to
contact_name = "G"

# Define the message to send
message = "Hello, how are you?"

# Open the Messages application
subprocess.call(
    ["osascript", "-e", 'tell application "{}" to activate'.format(app_name)]
)

# Wait for the Messages application to open
time.sleep(5)
# osascript -e 'tell application "System Events" to keystroke "Hello World"'

# Type the contact name into the search bar
subprocess.call(
    [
        "osascript",
        "-e",
        'tell application "System Events" to keystroke "{}"'.format(contact_name),
    ]
)
time.sleep(1)
subprocess.call(["osascript", "-e", 'tell application "System Events" to key code 36'])

# Wait for the conversation to load
time.sleep(5)

# Type the message into the text box
subprocess.call(
    [
        "osascript",
        "-e",
        'tell application "System Events" to keystroke "{}"'.format(message),
    ]
)
subprocess.call(["osascript", "-e", 'tell application "System Events" to key code 36'])

# Wait for the message to send
time.sleep(5)

# Check for a new message
new_message = subprocess.check_output(
    [
        "osascript",
        "-e",
        'tell application "{}" to get the text content of the last message of the first chat window whose name contains "{}"'.format(
            app_name, contact_name
        ),
    ]
)

# Print the new message
print(new_message.decode("utf-8"))

# def run_command(command):
# 		result subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
# 		return result.stdout.decode().strip()

# def send_message(phone_number, message):
# 		message= "Hello!"
# 		phone_number="+16175550133"
# 		subprocess.run(['osascript', '-e', f'tell application "Messages" to send "{message}" to buddy "{phone_number}"'])

# def get_last_message():
# 		output = run_command('osascript -e \'tell application "Messages" to set messagesList to get the last 1 messages of inbox 1\'')
# 		match = re.search('text "(.*)"', output)
# 		if match:
# 				return match.group(1)
# 		else:
# 				return None

# while True:
# 		last_message = get_last_message()
# 		if last_message:
# 				print(f'Got message: {last_message}')
# 				response = run_command(f'openai api completions.create --model <your-model> --prompt "{last_message}" --temperature 0.5 --max-tokens 60')
# 				response_text = response.split('"text": "')[1].split('", "')[0]
# 				print(f'Sending response: {response_text}')
# 				send_message('<phone-number>', response_text)
# 		else:
# 				print('No new messages.')
# 		time.sleep(5)
