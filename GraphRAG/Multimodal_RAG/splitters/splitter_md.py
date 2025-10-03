import base64
import hashlib
import io
import os
import re
from typing import List, Dict

from PIL import Image
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter

from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.milvus_db.db_operator import do_save_to_milvus
from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.my_llm import CustomQwen3Embeddings
from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.utils.common_utils import get_sorted_md_files
from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.utils.log_utils import log


class MarkdownDirSplitter:
    def __init__(self, images_output_dir: str, text_chunk_size: int = 1000):
        """

        :param images_output_dir: where the images are stored
        :param text_chunk_size:
        """
        self.images_output_dir = images_output_dir
        self.text_chunk_size = text_chunk_size
        os.makedirs(self.images_output_dir, exist_ok=True)

        # Title level configuration
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        self.text_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)

        # Split on semantic
        self.embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")
        self.semantic_splitter = SemanticChunker(
            self.embedding, breakpoint_threshold_type = "percentile"
        )

    def save_base64_to_image(self, base64_str: str, output_path: str) -> None:
        """将base64字符串解码为图像并保存"""
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        img.save(output_path)

    def process_images(self, content: str, source: str) -> List[Document]:
        """处理Markdown中的base64图片"""
        image_docs = []
        pattern = r'data:image/(.*?);base64,(.*?)\)'  # 正则匹配base64图片

        def replace_image(match):
            img_type = match.group(1).split(';')[0]
            base64_data = match.group(2)

            # 生成唯一文件名
            hash_key = hashlib.md5(base64_data.encode()).hexdigest()
            filename = f"{hash_key}.{img_type if img_type in ['png', 'jpg', 'jpeg'] else 'png'}"
            img_path = os.path.join(self.images_output_dir, filename)

            # 保存图片
            self.save_base64_to_image(base64_data, img_path)

            # 创建图片Document
            image_docs.append(Document(
                page_content=str(img_path),
                metadata={
                    "source": source,
                    "alt_text": "IMG",
                    "embedding_type": "image"
                }
            ))

            return "[IMG]"

        # 替换所有base64图片
        content = re.sub(pattern, replace_image, content, flags=re.DOTALL)
        return image_docs

    def remove_base64_images(self, text: str) -> str:
        """移除所有Base64图片标记"""
        pattern = r'!\[\]\(data:image/(.*?);base64,(.*?)\)'
        return re.sub(pattern, '', text)

    def process_md_file(self, md_file: str) -> List[Document]:
        """
        Process single md_file
        :param md_file:
        :return:
        """
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        # split the markdown content, split_documents is a list of texts and/or images
        split_documents: List[Document] = self.text_splitter.split_text(content)
        documents = []
        for doc in split_documents:
            # process image
            if '![](data:image/png;base64' in doc.page_content:
                image_docs: Document = self.process_images(doc.page_content, md_file)
                # The text content left after removing the images
                cleaned_content = self.remove_base64_images(doc.page_content)
                if cleaned_content.strip():
                    doc.metadata['embedding_type'] = 'text'
                    documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                documents.extend(image_docs)

            else:
                doc.metadata['embedding_type'] = 'text'
                documents.append(doc)

        # 语义分割
        final_docs = []
        for d in documents:
            if len(d.page_content) > self.text_chunk_size:
                final_docs.extend(self.semantic_splitter.split_documents([d]))
            else:
                final_docs.append(d)

        # 添加标题层级
        return final_docs

    def add_title_hierarchy(self, documents: List[Document], source_filename: str) -> List[Document]:
        """为文档添加标题层级结构"""
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []

        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata['source'] = source_filename

            # 更新标题状态
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key in new_metadata:
                    current_titles[level] = new_metadata[header_key]
                    for lower_level in range(level + 1, 4):
                        current_titles[lower_level] = ""

            # 补充缺失的标题
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key not in new_metadata:
                    new_metadata[header_key] = current_titles[level]
                elif current_titles[level] != new_metadata[header_key]:
                    new_metadata[header_key] = current_titles[level]

            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
            )

        return processed_docs

    def process_md_dir(self, md_dir: str, source_filename:str) -> List[Document]:
        """
        Specify a md directory, split all the data inside
        :param md_dir:
        :param source_filename: source file of md data (PDF)
        :return:
        """
        md_files = get_sorted_md_files(md_dir)
        all_documents = []
        for md_file in md_files:
            log.info(f"Processing {md_file}")
            all_documents.extend(self.process_md_file(md_file))

        return self.add_title_hierarchy(all_documents, source_filename)

if __name__ == "__main__":
    md_dir = r"/Users/lituokobe/Desktop/Mashibing_LLM_course/02_Live_Streaming2/Live_Streaming_practice2/GraphRAG/Multimodal_RAG/output/第一章 Apache Flink 概述"
    splitter = MarkdownDirSplitter(
        images_output_dir="/Users/lituokobe/Desktop/Mashibing_LLM_course/02_Live_Streaming2/Live_Streaming_practice2/GraphRAG/Multimodal_RAG/output/images",
    )
    docs = splitter.process_md_dir(md_dir, source_filename='第一章 Apache Flink 概述.pdf')

    res: List[Dict] = do_save_to_milvus(docs)

    for i, doc in enumerate(docs):
        print(f"\n文档 #{i + 1}:")
        print(doc)
        # print(f"内容: {doc.page_content[:30]}...")
        # print(f"元数据: {doc.metadata}...")
        #
        # print(f"一级标题: {doc.metadata.get('Header 1', '')}")
        # print(f"二级标题: {doc.metadata.get('Header 2', '')}")
        # print(f"三级标题: {doc.metadata.get('Header 3', '')}")
