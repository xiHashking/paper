import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class OpenAIClient:
    def __init__(self):
        """初始化OpenAI客户端"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未找到OpenAI API密钥。请在.env文件中设置OPENAI_API_KEY环境变量。")
        self.client = OpenAI(api_key=self.api_key)
    
    def chat_completion(self, messages, model="gpt-3.5-turbo", temperature=0.7):
        """
        使用OpenAI的Chat Completion API
        
        参数:
            messages (list): 消息列表，格式为[{"role": "user", "content": "你好"}]
            model (str): 使用的模型
            temperature (float): 温度参数，控制随机性
            
        返回:
            str: 模型回复的内容
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用OpenAI API时出错: {e}")
            return None
    
    def embeddings(self, text, model="text-embedding-3-small"):
        """
        获取文本嵌入向量
        
        参数:
            text (str): 要嵌入的文本
            model (str): 使用的嵌入模型
            
        返回:
            list: 嵌入向量
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取嵌入向量时出错: {e}")
            return None

# 测试代码
if __name__ == "__main__":
    client = OpenAIClient()
    response = client.chat_completion([
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下OpenAI。"}
    ])
    print(response) 