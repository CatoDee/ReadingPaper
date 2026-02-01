import arxiv
import pymupdf4llm
from openai import OpenAI
import os
import re
import time

# ================= 配置区域 =================
# 1. 填入你的 DeepSeek API Key
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# 2. 输入和输出文件设置
INPUT_FILE = "papers.txt"         # 存放 ArXiv 链接的文件 (一行一个)
OUTPUT_DIR = "paper_notes"        # 笔记保存目录

# 3. 字符数限制 (20万字符约等于 50k Tokens，足够覆盖 30-40 页的 ApJ 论文)
MAX_CHARS = 200000                 
# ===========================================

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def extract_arxiv_id(url):
    """从链接中提取 ArXiv ID"""
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    return match.group(1) if match else None

def strip_references(md_text):
    """
    尝试切除参考文献部分以节省 Token。
    pymupdf4llm 转换的 Markdown 通常会将 References 作为一级或二级标题。
    """
    # 常见的参考文献标题正则匹配
    # 匹配行首的 # References, ## Bibliography 等
    patterns = [
        r'\n#+\s*References', 
        r'\n#+\s*Bibliography', 
        r'\n#+\s*LITERATURE CITED'
    ]
    
    for pattern in patterns:
        # 搜索这些标题的位置
        matches = list(re.finditer(pattern, md_text, re.IGNORECASE))
        if matches:
            # 找到最后一个匹配项（防止目录中出现 References 字样误切）
            # 通常参考文献在文章最后，所以取最后一个匹配是比较安全的
            last_match = matches[-1]
            print(f"✂️  检测到参考文献部分，已切除 (位置: {last_match.start()}/{len(md_text)})")
            return md_text[:last_match.start()]
            
    return md_text

def get_paper_content(arxiv_id):
    """下载论文并转换为 Markdown"""
    print(f"⬇️  正在获取论文元数据: {arxiv_id}...")
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
    except Exception as e:
        raise Exception(f"ArXiv 下载失败: {e}")
    
    pdf_filename = f"{arxiv_id}.pdf"
    
    # 下载 PDF
    if not os.path.exists(pdf_filename):
        print(f"📥 正在下载 PDF...")
        paper.download_pdf(filename=pdf_filename)
    
    print(f"📖 正在解析 PDF (保留 LaTeX 公式)...")
    try:
        # 转换为 Markdown
        md_text = pymupdf4llm.to_markdown(pdf_filename)
        
        # 切除参考文献
        md_text = strip_references(md_text)
        
    except Exception as e:
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        raise Exception(f"PDF 解析失败: {e}")
    
    # 清理临时 PDF
    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)
    
    return paper.title, md_text

def analyze_with_deepseek(title, content):
    """调用 DeepSeek 进行深度总结"""
    print(f"🤖 DeepSeek 正在阅读并分析: {title}...")
    
    # 截断以防万一，虽然20w通常够用
    truncated_content = content[:MAX_CHARS]
    if len(content) > MAX_CHARS:
        print(f"⚠️  文章极长，已截取前 {MAX_CHARS} 字符")

    system_prompt = """
    你是一位资深的天体物理学研究员。请阅读用户提供的论文正文（Markdown格式）。
    
    【任务指令】
    1. 首先，请判断这篇论文的主要属性（单一或混合）：
       - **数值模拟 (Numerical Simulation)**
       - **天文观测 (Observational Astronomy)**
       - **理论推导 (Theoretical Astrophysics)**
    
    2. 请严格按照下方结构生成中文阅读报告。
    
    3. **关键要求**：
       - **保留 LaTeX 公式**：凡是涉及物理量（如 $\dot{M}$, $\Sigma_{gas}$, $\alpha_{vir}$）必须保留原格式。
       - **定量优先**：不要只说"结果增加"，要说"增加了约 3 倍"或"幂律指数为 -2.5"。
       - **代码与细节**：对于模拟，必须指出使用的 Code (e.g., ORION, ATHENA) 和关键算法。
    
    【输出结构】
    
    ### 1. 研究背景与动机 (Context & Motivation)
    - 研究对象（如：原恒星盘、分子云、星系反馈）。
    - 试图解决的具体物理张力 (Tension) 或 观测/理论 的缺失环节。
    
    ### 2. 研究方法 (Methodology)
    *(请根据论文类型智能调整重点)*
    - **[模拟]**：代码 (Code)、算法 (MHD/Hydro/PIC)、分辨率 (Resolution)、初始条件 (IC) 和 物理模块 (Physics modules)。
    - **[观测]**：望远镜 (Telescope)、观测波段 (Wavelength)、目标源 (Target)、灵敏度/波束大小 (Sensitivity/Beam)。
    - **[理论]**：核心假设 (Assumptions)、控制方程 (Governing Equations)、适用范围 (Regime)。
    
    ### 3. 主要结果 (Key Results)
    - **关键图表解读**：物理量的相关性（Correlations）或 演化趋势。
    - **数值结论**：提取文中的核心数值结果。
    - **模型验证**：模拟是否重现了观测？观测是否支持了理论？
    
    ### 4. 结论与讨论 (Conclusions & Discussion)
    - 核心物理图像 (Physical Picture) 的总结。
    - 局限性 (Caveats) 与 作者建议的未来工作 (Future Work)。
    """

    user_prompt = f"论文标题：{title}\n\n论文正文内容：\n{truncated_content}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2, # 保持低温，确保事实准确
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 接口调用出错: {e}"

def clean_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(INPUT_FILE):
        # 如果没有文件，自动创建一个示例
        with open(INPUT_FILE, "w") as f:
            f.write("# 在这里粘贴 ArXiv 链接，一行一个\n")
        print(f"❌ 找不到 {INPUT_FILE}，已自动创建。请填入链接后重试。")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total = len(urls)
    print(f"📋 发现 {total} 篇论文待处理...\n")

    for i, url in enumerate(urls):
        print(f"--- 处理第 {i+1}/{total} 篇 ---")
        try:
            arxiv_id = extract_arxiv_id(url)
            if not arxiv_id:
                print(f"⚠️ 跳过无效链接: {url}")
                continue
            
            # 1. 检查是否存在
            temp_search = arxiv.Search(id_list=[arxiv_id])
            try:
                temp_title = next(temp_search.results()).title
            except:
                temp_title = arxiv_id 
                
            safe_title = clean_filename(temp_title)
            output_path = os.path.join(OUTPUT_DIR, f"{safe_title}.md")
            
            if os.path.exists(output_path):
                print(f"✅ 笔记已存在，跳过: {safe_title}")
                continue

            # 2. 获取并清洗内容
            title, content = get_paper_content(arxiv_id)
            
            # 3. AI 分析
            report = analyze_with_deepseek(title, content)
            
            # 4. 保存
            with open(output_path, "w", encoding="utf-8") as f:
                header = f"# {title}\n\n**ArXiv ID**: [{arxiv_id}]({url})\n**Date**: {time.strftime('%Y-%m-%d')}\n\n---\n\n"
                f.write(header + report)
            
            print(f"✅ 报告已生成: {output_path}")
            
        except Exception as e:
            print(f"❌ 处理出错 {url}: {e}")
        
        print("\n")

if __name__ == "__main__":
    main()
