import re
import subprocess
import time

# Define the name of the application to open
app_name = "Messages"

# Define the contact name to send a message to
contact_name = "Bob Smith"

# Define the message to send
message = "Hello, how are you?"

# Open the Messages application
subprocess.call(
    ["osascript", "-e", 'tell application "{}" to activate'.format(app_name)]
)

# Wait for the Messages application to open
time.sleep(5)

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
        "osascript",Hello, how are you?

        "-e",
        'tell application "{}" to get the text content of the last message of the first chat window whose name contains "{}"'.format(
            app_name, contact_name
        ),
    ]
)

# Print the new message
print(new_message.decode("utf-8"))
