
import tkinter as tk
from tkinter import ttk
import sys

class ColoredText(tk.Text):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_tags()

    def configure_tags(self):
        self.tag_configure("USER", foreground="blue")
        self.tag_configure("ASSISTANT", foreground="green")

    def insert_message(self, message, tag):
        self.configure(state='normal')
        self.insert(tk.END, message + "\n", tag)
        self.configure(state='disabled')
        self.see(tk.END)

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert_message(string, "ASSISTANT")

    def flush(self):
        pass

def send_message(event=None):
    user_message = user_input.get()
    chatbox.insert_message("USER: " + user_message, "USER")
    user_input.delete(0, tk.END)
    print("ASSISTANT: " + "Sample response")  # Replace this with your response logic

root = tk.Tk()
root.title("Chat Panel")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

chatbox = ColoredText(main_frame, wrap="word", width=50, height=20, state='disabled')
chatbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=chatbox.yview)
scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
chatbox["yscrollcommand"] = scrollbar.set

user_input = ttk.Entry(main_frame, width=50)
user_input.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
user_input.bind("<Return>", send_message)

send_button = ttk.Button(main_frame, text="Send", command=send_message)
send_button.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

sys.stdout = RedirectText(chatbox)

root.mainloop()