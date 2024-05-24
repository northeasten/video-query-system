# 要保存的字符串
content = "这是一个测试字符串，用于保存到文件中。"

# 使用相对路径指定文件夹和文件路径
with open('.\\test\\test.txt', 'w', encoding='utf-8') as file:
    file.write(content)

print("文件已成功保存到: D:\\WorkSpace\\tasti\\test\\test.txt")
