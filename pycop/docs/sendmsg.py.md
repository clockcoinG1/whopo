# sendmsg.py			/Users/clockcoin/parsero/pycop/sendmsg.py
# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation to load before typing in message. 
 -  Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into the search bar using System Events keystroke command. 
    Then after one second delay, key code '36' (Enter) is pressed. 
-   After another five seconds wait period, text box in conversation window opens up where user types in their message
    using System Events keystroke command followed by pressing Enter key again with key code '36'.
-   Another five seconds wait period follows before checking for new messages which uses osascript's get last message content feature.

Is there anything else I can help you with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into search bar using System Events module. 
    Then after one second delay, enter key code (36) is pressed. 
    This loads conversation with Bob Smith in Messages app
-   Message text box gets focus and then Hello, how are you? message string gets typed into it using System Events module again. 
    Finally enter key code (36)is pressed again which sends this message
-   After another five seconds wait period new_message variable stores last received messages from Bob Smith's chat window.

Is there anything else I can help with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  A message variable containing the string "Hello, how are you?" is defined.
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into search bar using System Events keystroke function. 
    Then key code '36' (Enter) is pressed. 
-   After another wait of five seconds, text box in conversation window opens up and message variable content gets typed into it using System Events keystroke function again. Finally key code '36' (Enter)is pressed again.
-   Another wait of five seconds follows before checking for new messages by running an osascript command that retrieves last sent message from Bob Smith's chat window.

Is there anything else I can help with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into search bar using System Events module. 
    Then after one second delay, enter key code (36) is pressed. 
    This loads conversation with Bob Smith in Messages app
-   Message text box gets focus and then Hello, how are you? message string gets typed into it using System Events module again. 
    Finally enter key code (36)is pressed again which sends this message
-   After another wait of five seconds new_message variable stores last received messages from Bob Smith's chat window.

Is there anything else I can help with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  A message variable containing the string "Hello, how are you?" is defined.
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into search bar using System Events keystroke function. 
    Then key code '36' (Enter) is pressed. 
-   After another wait of five seconds, text box in conversation window opens up and message variable content gets typed into it using System Events keystroke function again. Finally key code '36' (Enter)is pressed again.
-   Another wait of five seconds follows before checking for new messages by running an osascript command that retrieves last sent message from Bob Smith's chat window.

Is there anything else I can help with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  The Messages application is opened using osascript command.
 -  After waiting for five seconds, the contact name is typed into search bar using System Events module. 
    Then after one second delay, enter key code (36) is pressed. 
    This loads conversation with Bob Smith in Messages app
-   Message text box gets focus and then Hello, how are you? message string gets typed into it using System Events module again. 
    Finally enter key code (36)is pressed again which sends this message
-   After another wait of five seconds new_message variable stores last received messages from Bob Smith's chat window.

Is there anything else I can help with?

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

# sendmsg.py

## Summary

The file imports the re, subprocess and time modules.
 -  The name of the application to open is defined as "Messages".
 -  The contact name to send a message to is defined as "Bob Smith".
 -  The message to send is defined as "Hello, how are you?".
 -  It opens the Messages application using osascript command.
 -  It types in the contact name into search bar and waits for conversation window to load before typing in message. 
 - Finally it checks for new messages and prints them out.

## Code

```python
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

```

## Filepath

```/Users/clockcoin/parsero/pycop/sendmsg.py```

