import tempfile
from share_btn import community_icon_html, loading_icon_html, share_js, save_js
import huggingface_hub
import gradio as gr
# from gill import utils
# from gill import models
# 没有用到呀
# import evals.utils as utils
# import evals.models as models
import utils
import models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
# HF Transfer 功能允许用户在 Hugging Face Hub 上缓存模型，并在需要时自动从缓存中加载模型。
# 禁用这个功能意味着在加载模型时，不会使用 HF Transfer 功能，而是直接从远程下载模型
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "False"

# #save-btn、#share-btn 分别设置了 id 为 save-btn 和 share-btn 的按钮元素的背景渐变颜色。
css = """
    # 设置 id 为 chatbot 的元素的最小高度为 300px，确保聊天机器人界面的高度不会小于 300px。
    #chatbot { min-height: 300px; }
    # #save-btn、#share-btn 分别设置了 id 为 save-btn 和 share-btn 的按钮元素的背景渐变颜色。
    # 当鼠标悬停在按钮上时，背景颜色会发生变化，采用不同的渐变颜色。
    #save-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #save-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    #share-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #share-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    # 当鼠标悬停在按钮图片上时，取消图片的放大效果并恢复正常大小。
    #gallery { z-index: 999999; }
    #gallery img:hover {transform: scale(2.3); z-index: 999999; position: relative; padding-right: 30%; padding-bottom: 30%;}
    #gallery button img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; padding-bottom: 0;}
    # 在不支持鼠标悬停的设备上，取消图片的放大效果（transform: none），恢复到原始大小。
    @media (hover: none) {
        #gallery img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; 0;}
    }
    .html2canvas-container { width: 3000px !important; height: 3000px !important; }
"""

examples = [
    'examples/car.png',
    'examples/cake.png',
    'examples/house.png',
    'examples/maple leaf.png',
    'examples/train.png',
]

# Download model from HF Hub.
# ckpt_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='pretrained_ckpt.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# decision_model_path = huggingface_hub.hf_hub_download(
#     repo_id='jykoh/gill', filename='decision_model.pth.tar', local_dir='/data1/ruip/gill/gill-main_1/download')
# args_path = huggingface_hub.hf_hub_download(
#    repo_id='jykoh/gill', filename='model_args.json', local_dir='/data1/ruip/gill/gill-main_1/download')
path = '/data1/ruip/gill/gill-main_1/checkpoints/gill_opt'
# model = models.load_gill('./', args_path, ckpt_path, decision_model_path)
model = models.load_gill(path)
print("models load complished\n")
# 接受两个参数 state 和 image_input
# state 是一个包含两个元素的列表，分别表示对话和聊天历史记录；image_input 是一个上传的图像文件对象
def upload_image(state, image_input):
    conversation = state[0]
    chat_history = state[1]
    # 使用 PIL 库中的 Image.open() 方法打开上传的图像文件，
    # 然后调用 resize() 方法将图像大小调整为 (224, 224) 像素，再调用 convert('RGB') 方法将图像转换为 RGB 模式
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    # 保存处理后的图像，覆盖原图像文件。这一步可能是为了减少图像尺寸，以节省存储空间或加快处理速度
    input_image.save(image_input.name)  # Overwrite with smaller image.
    # 处理后的图像路径以 <img> 标签的形式添加到对话记录中，该标签用于将图像显示在对话界面中
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    # 返回更新后的状态和对话记录。l
    return [conversation, chat_history + [input_image, ""]], conversation


def reset():
    return [[], []], []


def reset_last(state):
    # 使用切片操作 [:-1]，将最后一条对话从对话记录中删除，得到更新后的对话记录 conversation
    conversation = state[0][:-1]
    # 使用切片操作 [:-2]，将最后两条聊天历史记录从聊天历史中删除，得到更新后的聊天历史记录 chat_history
    chat_history = state[1][:-2]
    return [conversation, chat_history], conversation


def save_image_to_local(image: Image.Image):
    # TODO(jykoh): Update so the url path is used, to prevent repeat saving.
    # 更新代码，使用 URL 路径来保存图像，避免重复保存
    # tempfile._get_candidate_names() 是 Python 中 tempfile 模块内部的一个函数，主要用于生成临时文件名的候选列表
    # 使用 next() 函数获取其中的第一个文件名
    filename = next(tempfile._get_candidate_names()) + '.png'
    # 将图像 image 保存到这个生成的文件名中
    image.save(filename)
    return filename

# 输入：
# 输入文本
# state: 表示模型的当前状态，可能包含了模型内部的一些信息或者状态
# ret_scale_factor: 表示返回的缩放因子，用于调整生成文本的返回长度
# num_words: 表示生成文本的长度或者单词数目
# temperature: 表示控制生成文本的多样性和创造性的温度参数。较高的温度会导致更加随机和多样化的生成结果，而较低的温度则会倾向于生成更加确定性和传统的文本
def generate_for_prompt(input_text, state, ret_scale_factor, num_words, temperature):
    # torch.Generator 类来创建一个生成器对象 g_cuda，并指定该生成器在 CUDA 设备上进行操作，并手动设置种子为 1337
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    # Ignore empty inputs.
    if len(input_text) == 0:
        return state, state[0], gr.update(visible=True)
    
    input_prompt = 'Q: ' + input_text + '\nA:'
    conversation = state[0]
    chat_history = state[1]
    print('Generating for', chat_history, flush=True)

    # If an image was uploaded, prepend it to the model. 如果上传了一张图片，将其添加到模型前面
    # 这里我的理解是chat_history其实就是上传的图片，历史的聊天记录是图片
    model_inputs = chat_history
    model_inputs.append(input_prompt)
    # Remove empty text.
    model_inputs = [s for s in model_inputs if s != '']
    # 存在一定的温度调节生成文本的情况下，top_p 的值被调整为 0.95。这可以影响模型生成文本时对词汇的采样方式，进而影响生成文本的多样性和创造性。
    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95

    print('Running model.generate_for_images_and_texts with', model_inputs, flush=True)
    # model在哪里
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=1,
                                                        num_inference_steps=50, generator=g_cuda)
    print('model_outputs', model_outputs, ret_scale_factor, flush=True)

    response = ''
    text_outputs = []
    for output_i, p in enumerate(model_outputs):
        if type(p) == str:
            if output_i > 0:
                response += '<br/>'
            # Remove the image tokens for output.
            text_outputs.append(p.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', ''))
            response += p
            if len(model_outputs) > 1:
                response += '<br/>'
        elif type(p) == dict:
            # Decide whether to generate or retrieve.
            if p['decision'] is not None and p['decision'][0] == 'gen':
                image = p['gen'][0][0]#.resize((224, 224))
                filename = save_image_to_local(image)
                response += f'<img src="./file={filename}" style="display: inline-block;"><p style="font-size: 12px; color: #555; margin-top: 0;">(Generated)</p>'
            else:
                image = p['ret'][0][0]#.resize((224, 224))
                filename = save_image_to_local(image)
                response += f'<img src="./file={filename}" style="display: inline-block;"><p style="font-size: 12px; color: #555; margin-top: 0;">(Retrieved)</p>'

    chat_history = model_inputs + \
        [' '.join([s for s in model_outputs if type(s) == str]) + '\n']
    # Remove [RET] from outputs.
    conversation.append((input_text, response.replace('[IMG0] [IMG1] [IMG2] [IMG3] [IMG4] [IMG5] [IMG6] [IMG7]', '')))

    # Set input image to None.
    print('state', state, flush=True)
    print('updated state', [conversation, chat_history], flush=True)
    return [conversation, chat_history], conversation, gr.update(visible=True), gr.update(visible=True)

# 创建一个gr.Blocks对象，用于包含交互式组件布局
with gr.Blocks(css=css) as demo:
    # gr.HTML(""" ... """)：在 Blocks 中添加 HTML 内容，用于展示一些静态文本和链接
    gr.HTML("""
        <h1>🐟 GILL</h1>
        <p>This is the official Gradio demo for the GILL model, a model that can process arbitrarily interleaved image and text inputs, and produce image and text outputs.</p>

        <strong>Paper:</strong> <a href="https://arxiv.org/abs/2305.17216" target="_blank">Generating Images with Multimodal Language Models</a>
        <br/>
        <strong>Project Website:</strong> <a href="https://jykoh.com/gill" target="_blank">GILL Website</a>
        <br/>
        <strong>Code and Models:</strong> <a href="https://github.com/kohjingyu/gill" target="_blank">GitHub</a>
        <br/>
        <br/>

        <strong>Tips:</strong>
        <ul>
        <li>Start by inputting either image or text prompts (or both) and chat with GILL to get image-and-text replies.</li>
        <li>Tweak the level of sensitivity to images and text using the parameters on the right.</li>
        <li>Check out cool conversations in the examples or community tab for inspiration and share your own!</li>
        <li>If the model outputs a blank image, it is because Stable Diffusion's safety filter detected inappropriate content. Please try again with a different prompt.</li>
        <li>Outputs may differ slightly from the paper due to slight implementation differences. For reproducing paper results, please use our <a href="https://github.com/kohjingyu/gill" target="_blank">official code</a>.</li>
        <li>For faster inference without waiting in queue, you may duplicate the space and use your own GPU: <a href="https://huggingface.co/spaces/jykoh/gill?duplicate=true"><img style="display: inline-block; margin-top: 0em; margin-bottom: 0em" src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></li>
        </ul>
    """)
    # 创建一个状态对象，用于跟踪对话内容和聊天历史
    gr_state = gr.State([[], []])  # conversation, chat_history
    # 创建一个行布局，在这个布局中放置多个交互式组件。
    with gr.Row():
        # 创建一个列布局，设置比例和最小宽度
        with gr.Column(scale=0.7, min_width=500):
            with gr.Row():
                # 创建一个聊天机器人组件，并设置标签
                chatbot = gr.Chatbot(elem_id="chatbot", label="🐟 GILL Chatbot")
            with gr.Row():
                # 创建一个上传按钮组件，用于上传图片文件
                image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                # 创建一个文本框组件，用于用户输入信息
                text_input = gr.Textbox(label="Message", placeholder="Type a message")
                with gr.Column():
                    submit_btn = gr.Button(
                        "Submit", interactive=True, variant="primary")
                    clear_last_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Reset All")
                    with gr.Row(visible=False) as save_group:
                        save_button = gr.Button("💾 Save Conversation as .png", elem_id="save-btn")

                    with gr.Row(visible=False) as share_group:
                        share_button = gr.Button("🤗 Share to Community (opens new window)", elem_id="share-btn")

        with gr.Column(scale=0.3, min_width=400):
            ret_scale_factor = gr.Slider(minimum=0.0, maximum=3.0, value=1.3, step=0.1, interactive=True,
                                         label="Frequency multiplier for returning images (higher means more frequent)")
            gr_max_len = gr.Slider(minimum=1, maximum=64, value=32,
                                   step=1, interactive=True, label="Max # of words")
            gr_temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature (0 for deterministic, higher for more randomness)")

#            gallery = gr.Gallery(
#                value=[Image.open(e) for e in examples], label="Example Conversations", show_label=True, elem_id="gallery",
#            ).style(grid=[2], height="auto")
            gallery = gr.Gallery(
                value=[Image.open(e) for e in examples], label="Example Conversations", show_label=True, elem_id="gallery",
            )
    text_input.submit(generate_for_prompt, [text_input, gr_state, ret_scale_factor,
                      gr_max_len, gr_temperature], [gr_state, chatbot, share_group, save_group])
    text_input.submit(lambda: "", None, text_input)  # Reset chatbox.
    submit_btn.click(generate_for_prompt, [text_input, gr_state, ret_scale_factor,
                     gr_max_len, gr_temperature], [gr_state, chatbot, share_group, save_group])
    submit_btn.click(lambda: "", None, text_input)  # Reset chatbox.

    image_btn.upload(upload_image, [gr_state, image_btn], [gr_state, chatbot])
    clear_last_btn.click(reset_last, [gr_state], [gr_state, chatbot])
    clear_btn.click(reset, [], [gr_state, chatbot])
#    share_button.click(None, [], [], _js=share_js)
#    save_button.click(None, [], [], _js=save_js)
    share_button.click(None, [], [])
    save_button.click(None, [], [])


demo.queue(concurrency_count=1, api_open=False, max_size=16)
demo.launch(debug=True, server_name="0.0.0.0", share=False)
