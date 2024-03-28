from transformers import CLIPTextModel, CLIPTokenizer


# 文本编码
def prompts_embedding(prompts):
    # 加载编码模型
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # tokenizer.model_max_length -> 77
    text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids)
    text_embeddings = text_embeddings[0]  # (1, 77, 768)

    return text_embeddings


def test_embedding():
    prompts = ["a photograph of an astronaut riding a horse"]
    text_embeddings = prompts_embedding(prompts)

    uncond_prompts = [""]
    uncond_embeddings = prompts_embedding(uncond_prompts)

    print("text_embeddings.shape", text_embeddings.shape)
    print("text_embeddings.shape", uncond_embeddings.shape)


test_embedding()