import logging
import streamlit as st
from typing import Optional, List, Dict
import dashscope
import os
import pandas as pd
from io import StringIO
import json
import warnings

# Suppress NumPy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

# 本地Qwen依赖
import mindspore as ms
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread
from mindspore import ops  # 添加导入ops

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 设置阿里云DashScope API密钥（请替换为您的实际密钥）
os.environ["DASHSCOPE_API_KEY"] = "sk-a1e20166519143fdabfc03d4b38ba595"  # 在实际使用中，从环境变量或安全方式获取

class QwenAnalyzer:
    """使用阿里云Qwen模型的多任务分析器（云端）"""
    def __init__(self, model: str = "qwen-turbo"):  # 可以选择qwen-turbo, qwen-plus 等
        self.model = model
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")

    def analyze(self, text: str, task: str) -> Optional[str]:
        """通用分析方法，根据任务构建提示并调用Qwen API"""
        prompts = {
            "summary": f"为下面的文本生成摘要：\n{text}",
            "sentiment": f"分析下面的文本情感倾向（积极、中性、消极）：\n{text}",
            "keywords": f"从下面的文本中抽取关键词（用逗号分隔）：\n{text}",
            "intent": f"分类下面的文本意图（投诉、赞扬、建议、其他）：\n{text}",
            "profile": f"基于多条评论生成用户画像（兴趣、情感倾向、总结）：\n{text}"  # 对于用户画像，输入多条评论拼接
        }
        prompt = prompts.get(task, f"处理下面的文本：\n{text}")
        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=200,  # 控制输出长度
                temperature=0.7,  # 调整创造性
                top_p=1.0
            )
            if response.status_code == 200:
                output = response.output.text.strip()
                logging.info(f"{task} 结果: {output}")
                return output
            else:
                logging.error(f"API 调用失败: {response.message}")
                return None
        except Exception as e:
            logging.error(f"{task} 推理失败: {e}")
            return None

class LocalQwenAnalyzer:
    """使用本地Qwen模型的多任务分析器"""
    def __init__(self, model_path: str = "Qwen/Qwen1.5-0.5B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, dtype=ms.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=ms.float16)
        self.system_prompt = "You are an expert in text analysis."

    def analyze(self, text: str, task: str) -> Optional[str]:
        prompts = {
            "summary": f"为下面的文本生成摘要：\n{text}",
            "sentiment": f"分析下面的文本情感倾向（积极、中性、消极）：\n{text}",
            "keywords": f"从下面的文本中抽取关键词（用逗号分隔）：\n{text}",
            "intent": f"分类下面的文本意图（投诉、赞扬、建议、其他）：\n{text}",
            "profile": f"基于多条评论生成用户画像（兴趣、情感倾向、总结）：\n{text}"
        }
        user_msg = prompts.get(task, f"处理下面的文本：\n{text}")
        messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': user_msg}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="ms",
            tokenize=True
        )
        
        # 创建attention_mask来避免警告
        attention_mask = ops.ones(input_ids.shape, dtype=ms.int64)
        
        streamer = TextIteratorStreamer(self.tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 添加attention_mask
            streamer=streamer,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            if '</s>' in partial_message:
                break
        output = partial_message.strip()
        logging.info(f"{task} 结果: {output}")
        return output

def parse_user_comments(uploaded_file) -> List[Dict]:
    """解析单个用户TXT：主题：XXX 时间：XXX 内容：XXX"""
    content = uploaded_file.getvalue().decode("utf-8")
    comments = []
    lines = content.splitlines()
    current = {}
    for line in lines:
        if line.startswith("主题："):
            if current: comments.append(current)
            current = {"theme": line[3:].strip()}
        elif line.startswith("时间："):
            current["time"] = line[3:].strip()
        elif line.startswith("内容："):
            current["content"] = line[3:].strip()
    if current: comments.append(current)
    return comments

def parse_theme_comments(uploaded_file) -> List[Dict]:
    """解析主题TXT：主题：XXX 用户：XXX 时间：XXX 内容：XXX"""
    content = uploaded_file.getvalue().decode("utf-8")
    comments = []
    lines = content.splitlines()
    current = {}
    for line in lines:
        if line.startswith("主题："):
            if current: comments.append(current)
            current = {"theme": line[3:].strip()}
        elif line.startswith("用户："):
            current["user"] = line[3:].strip()
        elif line.startswith("时间："):
            current["time"] = line[3:].strip()
        elif line.startswith("内容："):
            current["content"] = line[3:].strip()
    if current: comments.append(current)
    return comments

# ======================================================================
# Streamlit UI
# ======================================================================

st.set_page_config(page_title="舆情监控系统", layout="wide")

st.title("舆情监控系统（Qwen·增强版）")
st.sidebar.title("导航")

analysis_type = st.sidebar.radio("选择分析类型", ("单个用户分析", "主题分析", "图像角色分析"))

# 根据分析类型调整模式选择
if analysis_type == "图像角色分析":
    mode = "云端 (阿里云Qwen)"  # 强制云端模式
    st.sidebar.info("🖼️ 图像分析仅支持云端模式")
else:
    mode = st.sidebar.radio("选择分析模式", ("云端 (阿里云Qwen)", "本地（Qwen0.5B）"))

# 选择模型
if mode.startswith("云端"):
    analyzer = QwenAnalyzer()
else:
    analyzer = LocalQwenAnalyzer() 

# Set MindSpore context with fallback
try:
    ms.set_context(device_target='Ascend', device_id=0, mode=ms.PYNATIVE_MODE)
    st.info("Using Ascend device.")
except Exception as e:
    st.warning(f"Ascend device not available: {e}. Falling back to CPU.")
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

# 根据模式初始化分析器
if mode == "云端 (阿里云Qwen)":
    analyzer = QwenAnalyzer(model="qwen-turbo")
else:
    analyzer = LocalQwenAnalyzer()

if analysis_type == "单个用户分析":
    st.header("单个用户分析")
    username = st.text_input("输入用户名")
    uploaded_file = st.file_uploader("上传用户评论TXT文件", type="txt")
    
    if uploaded_file and username:
        comments = parse_user_comments(uploaded_file)
        st.subheader(f"用户 {username} 的评论列表")
        df = pd.DataFrame(comments)
        st.dataframe(df)
        
        # 分析每条
        st.subheader("每条评论分析")
        results = []
        for i, comment in enumerate(comments):
            content = comment['content']
            sentiment = analyzer.analyze(content, "sentiment")
            keywords = analyzer.analyze(content, "keywords")
            intent = analyzer.analyze(content, "intent")
            summary = analyzer.analyze(content, "summary")
            theme = comment.get('theme', '未知')
            time = comment.get('time', '未知')
            st.markdown(f"**评论 {i+1} ({theme}, {time})**")
            st.write(f"情感: {sentiment}")
            st.write(f"关键词: {keywords}")
            st.write(f"意图: {intent}")
            st.write(f"摘要: {summary}")
            results.append({
                "comment_id": i+1,
                "theme": theme,
                "time": time,
                "content": content,
                "sentiment": sentiment,
                "keywords": keywords,
                "intent": intent,
                "summary": summary
            })
        
        # 用户画像
        all_content = "\n".join([c['content'] for c in comments])
        profile = analyzer.analyze(all_content, "profile")
        st.subheader("用户画像")
        st.write(profile)
        results.append({"profile": profile})
        
        # 保存按钮
        json_results = json.dumps(results, ensure_ascii=False, indent=4)
        st.download_button(
            label="下载分析结果",
            data=json_results,
            file_name=f"user_{username}_analysis.json",
            mime="application/json"
        )

elif analysis_type == "主题分析":
    st.header("主题分析")
    theme_input = st.text_input("输入主题")
    uploaded_file = st.file_uploader("上传主题评论TXT文件", type="txt")
    
    if uploaded_file and theme_input:
        comments = parse_theme_comments(uploaded_file)
        st.subheader(f"主题 {theme_input} 的评论列表")
        df = pd.DataFrame(comments)
        st.dataframe(df)
        
        # 分析每条
        st.subheader("每条评论分析")
        sentiments = []
        results = []
        for i, comment in enumerate(comments):
            content = comment['content']
            sentiment = analyzer.analyze(content, "sentiment")
            keywords = analyzer.analyze(content, "keywords")
            intent = analyzer.analyze(content, "intent")
            summary = analyzer.analyze(content, "summary")
            user = comment.get('user', '未知')
            time = comment.get('time', '未知')
            st.markdown(f"**评论 {i+1} ({user}, {time})**")
            st.write(f"情感: {sentiment}")
            st.write(f"关键词: {keywords}")
            st.write(f"意图: {intent}")
            st.write(f"摘要: {summary}")
            sentiments.append(sentiment)
            results.append({
                "comment_id": i+1,
                "user": user,
                "time": time,
                "content": content,
                "sentiment": sentiment,
                "keywords": keywords,
                "intent": intent,
                "summary": summary
            })
        
        # 整体总结
        all_content = "\n".join([c['content'] for c in comments])
        overall_summary = analyzer.analyze(all_content, "summary")
        sentiment_dist = pd.Series(sentiments).value_counts().to_dict()
        st.subheader("整体总结")
        st.write(f"情感分布: {sentiment_dist}")
        st.write(f"总体摘要: {overall_summary}")
        results.append({
            "sentiment_distribution": sentiment_dist,
            "overall_summary": overall_summary
        })
        
        # 保存按钮
        json_results = json.dumps(results, ensure_ascii=False, indent=4)
        st.download_button(
            label="下载分析结果",
            data=json_results,
            file_name=f"theme_{theme_input}_analysis.json",
            mime="application/json"
        )

# 运行提示
if __name__ == "__main__":
    st.write("应用已加载。选择导航和模式开始分析。")

# ======================================================================
# 新功能：图像角色分析
# ======================================================================

elif analysis_type == "图像角色分析":

    st.header("图像角色分析")
    uploaded_image = st.file_uploader("🖼️ 上传图片文件", type=["jpg", "jpeg", "png"], help="支持 JPG/PNG 格式，上传后自动分析")

    if uploaded_image:
        st.image(uploaded_image, caption="上传的图片", width=400)  # 控制尺寸为400px

        # 读取图片为 base64
        image_data = uploaded_image.getvalue()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        mime_type = uploaded_image.type  # e.g., image/jpeg

        # Prompt
        prompt = "你是一名有着多年刑侦经验和断案经验的警察，你尤其对人物肖像敏感，你可以轻松的判断出画面中的人的年龄、职业、性格等等各项身份信息，给出你的分析。"

        # 多模态调用
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:{mime_type};base64,{base64_image}"},
                    {"text": prompt}
                ]
            }
        ]

        try:
            with st.spinner("🔍 正在分析图片..."):
                response = dashscope.MultiModalConversation.call(
                    model="qwen-vl-plus",
                    messages=messages
                )
            if response.status_code == 200:
                analysis_result = response.output.choices[0].message.content[0]["text"]
                st.subheader("分析结果")
                st.write(analysis_result)

                # 下载按钮
                results = {"analysis": analysis_result}
                st.download_button(
                    "📥 下载分析结果",
                    json.dumps(results, ensure_ascii=False, indent=4),
                    "image_analysis.json",
                    "application/json"
                )
            else:
                st.error(f"API 调用失败: {response.message}")
        except Exception as e:
            st.error(f"调用失败: {e}")

if __name__ == "__main__":
    st.write("应用已加载。选择导航和模式开始分析。")