import tkinter as tk
from tkinter import messagebox

# Function to handle user input and respond based on predefined rules
def chatbot_response():
    user_input = user_entry.get().lower()
    response = ""

    # Predefined responses based on user queries
    if "hello" in user_input or "hi" in user_input:
        response = "Hello! How can I assist you today?"
    elif "your name" in user_input:
        response = "I'm ChatBot, your virtual assistant."
    elif "how are you" in user_input:
        response = "I'm here to help! How can I assist you?"
    elif "bye" in user_input or "exit" in user_input:
        response = "Goodbye! Have a great day!"
    elif "thank you" in user_input or "thanks" in user_input:
        response = "You're welcome! I'm here to help anytime."
    else:
        response = "Sorry, I didn't understand that. Could you rephrase?"

    # Display response in the chat box
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "You: " + user_input + "\n")
    chat_box.insert(tk.END, "Bot: " + response + "\n\n")
    chat_box.config(state=tk.DISABLED)
    user_entry.delete(0, tk.END)

# Main window
window = tk.Tk()
window.title("ChatBot Interface")
window.geometry("400x500")
window.config(bg="#222831")

# Header label
header_label = tk.Label(window, text="AI ChatBot", bg="#222831", fg="#00ADB5", font=("Helvetica", 16, "bold"))
header_label.pack(pady=10)

# Chat display box
chat_box = tk.Text(window, bd=1, bg="#393E46", fg="#EEEEEE", font=("Arial", 12), wrap="word", state=tk.DISABLED)
chat_box.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

# Scrollbar for chat box
scrollbar = tk.Scrollbar(chat_box)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_box.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=chat_box.yview)

# User input field
user_entry = tk.Entry(window, bg="#222831", fg="#EEEEEE", font=("Arial", 12))
user_entry.pack(pady=10, padx=10, fill=tk.X)

# Send button
send_button = tk.Button(window, text="Send", bg="#00ADB5", fg="#EEEEEE", font=("Arial", 12), command=chatbot_response)
send_button.pack(pady=10)

# Run main loop
window.mainloop()