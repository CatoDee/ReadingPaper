# ReadingPaper
一个用于阅读arxiv论文的小工具。可以先将自己感兴趣的论文arxiv链接复制到papers.txt文件中，然后用python运行代码，它会将论文喂给deepseek然后每篇论文出报告。

我是配合benty-fields使用的。先自己刷一遍标题和摘要，再用AI读一下，如果有必要那就再详细的阅读。tokens应该能覆盖30页的APJ，略过了参考文献部分。

## python需要的包
pip install arxiv openai pymupdf4llm
