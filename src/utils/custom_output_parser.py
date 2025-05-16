from langchain.schema import BaseOutputParser

class CustomListOutputParser(BaseOutputParser):
    separator: str = "\n"

    def __init__(self, separator="\n"):
        super().__init__()
        self.separator = separator
    
    def parse(self, text: str) -> list:
        items = text.strip().split(self.separator)
        return [item.strip() for item in items if item.strip()]
    
    def get_format_instructions(self) -> str:
        if self.separator == "\n":
            return "Hãy liệt kê các mục, mỗi mục trên một dòng riêng biệt."
        else:
            return f"Hãy liệt kê các mục, phân tách bằng '{self.separator}'."