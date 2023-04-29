import shutil, os, subprocess, sys, time, json, re, argparse
import osascript
import os
import xml.etree.ElementTree as ET

# Locate the Shortcuts.sdef file
shortcuts_sdef_path = "/System/Applications/Shortcuts.app/Contents/Resources/Shortcuts.sdef"
# Check if the file exists
if not os.path.exists(shortcuts_sdef_path):
    print("Shortcuts.sdef file not found.")
    exit()
# Parse the Shortcuts.sdef file
try:
    tree = ET.parse(shortcuts_sdef_path)
    root = tree.getroot()
except:
    print("Error parsing Shortcuts.sdef file.")
    exit()
# Extract the required information into a dictionary
for elem in root.iter():
    print("-" * 50)
    for x in elem.keys():
        if elem.get(x):
            print(f"{x} : {elem.get(x)}")
        else:
            print(f"{x} : None")


def run_shortcut(shortcut_name, input_data=None):
    input_param = f'with input "{input_data}"' if input_data else ''
    code = f'''
    tell application "Shortcuts Events"
        set the_shortcut to shortcut named "{shortcut_name}"
        run the_shortcut {input_param}
    end tell
    '''
    return osascript.run(code)


def get_shortcut_info(shortcut_name):
    code = f'''
    tell application "Shortcuts Events"
        set the_shortcut to shortcut named "{shortcut_name}"
        set shortcut_info to {{name: name of the_shortcut, subtitle: subtitle of the_shortcut, id: id of the_shortcut, folder: name of folder of the_shortcut, color: color of the_shortcut, icon: icon of the_shortcut, accepts_input: accepts input of the_shortcut, action_count: action count of the_shortcut}}
    end tell
    return shortcut_info
    '''
    return osascript.run(code)


def get_folder_info(folder_name):
    code = f'''
    tell application "Shortcuts"
        set the_folder to folder named "{folder_name}"
        set folder_info to {{name: name of the_folder, id: id of the_folder, shortcuts: shortcuts of the_folder}}
    end tell
    return folder_info
    '''
    return osascript.run(code)


# Example usage


def main():
    shortcut_name = "chatGPT"
    folder_name = "~/Library/Shortcuts"

    # Run a shortcut with input
    result = run_shortcut(shortcut_name, os.execvp("pbpaste", ["pbpaste", "-Prefer", "txt"]))
    print(result)

    # Get shortcut info
    shortcut_info = get_shortcut_info(shortcut_name)
    print(shortcut_info)

    # Get folder info
    folder_info = get_folder_info(folder_name)
    print(folder_info)
