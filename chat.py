from openai import OpenAI
# wraps 
class ChatWithMemory(OpenAI):
    def __init__(self, organization, project, chat_memory):
        super().__init__(organization, project)
        self.chat_memory = chat_memory
    def send_message(self, message):
        self.chat_memory.append(message)
        return super().send_message(message)
    def send_messages(self, messages):
        self.chat_memory.extend(messages)
        return super().send_messages(messages)
    def reset(self):
        self.chat_memory.clear()
        return super().reset()
    def get_chat_memory(self):
        return self.chat_memory
    def set_chat_memory(self, chat_memory):
        self.chat_memory = chat_memory