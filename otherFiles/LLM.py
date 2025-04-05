import openai


def query_llm(mirna, gnn_score, known_diseases, model="gpt-4o"):
    """
    询问 LLM 预测 miRNA 是否与某疾病相关
    :param mirna: miRNA 名称
    :param gnn_score: GNN 预测的概率分数
    :param known_diseases: 该 miRNA 已知相关的疾病列表
    :param model: 选择的 LLM 模型
    :return: LLM 预测的概率和置信度
    """
    prompt = f"""
    已知 miRNA: {mirna}
    GNN 预测其与某疾病的关联概率：{gnn_score:.4f}
    该 miRNA 已知相关疾病：{", ".join(known_diseases) if known_diseases else "未知"}
    请基于已有生物知识推测该 miRNA 是否可能与该疾病相关？
    请以 JSON 格式返回，例如：
    {{
        "LLM_Prediction": "是",
        "Confidence": 0.75
    }}
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # 控制 LLM 生成的随机性
    )

    llm_output = response["choices"][0]["message"]["content"]
    return eval(llm_output)  # 解析 JSON
