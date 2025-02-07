if __name__ == '__main__':

    import os
    import argparse
    import pandas as pd
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    from latb.evaluate import AverageTokenLength, LangConfusion, RougeEval
    
    parser = argparse.ArgumentParser(
        description='Please provide the folder path for evaluation'
    )

    parser.add_argument('-f', '--folder', required=True)
    args = vars(parser.parse_args())
    folder_path = args['folder']
    folder_name = os.path.basename(folder_path)
    print(f"Evaluating folder {folder_name}")
    file_names = os.listdir(folder_path)

    lang_code_map = {
        "arabic": "ar",
        "chinese": "zh",
        "english": "en",
        "french": "fr",
        "hindi": "hi",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "thai": "th"
    }

    df = {
        "file_name": [],
        "lang_code": [],
        "avg_token_length": [],
        "token_level_confusion": [],
        "line_level_confusion": [],
        "response_level_confusion": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }

    # paths
    load_dotenv(dotenv_path="../../.env")
    tokenizer_path = os.getenv("LLAMA3_PATH")
    language_id_path = os.getenv("FASTTEXT_PATH")

    # evaluation loop
    for i, file_name in enumerate(file_names):
        print(f"Evaluating {file_name}...")
        absolute_file_path = os.path.join(folder_path, file_name)

        # file name and language
        df["file_name"].append(file_name)
        lang = file_name.split("_")[0].split("-")[-1]
        target_lang_id = lang_code_map[lang]
        df["lang_code"].append(target_lang_id)

        # average token length
        avg_tok_len = AverageTokenLength(tokenizer_path=tokenizer_path)
        res_avg_tok_len = avg_tok_len.eval(file_path=absolute_file_path)
        df["avg_token_length"].append(res_avg_tok_len)

        # language confusion
        lang_confusion = LangConfusion(
            tokenizer_path=tokenizer_path,
            language_id_path=language_id_path
        )
        res_token_level, _ = lang_confusion.eval_token_level_unicode(
            file_path=absolute_file_path, 
            target_lang_id=target_lang_id
        )
        df["token_level_confusion"].append(res_token_level*100)

        res_line_level, _ = lang_confusion.eval_line_level_fasttext(
            file_path=absolute_file_path, 
            target_lang_id=target_lang_id
        )
        df["line_level_confusion"].append(res_line_level*100)

        res_response_level, _ = lang_confusion.eval_response_level_fasttext(
            file_path=absolute_file_path, 
            target_lang_id=target_lang_id
        )
        df["response_level_confusion"].append(res_response_level*100)

        # rouge scores
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer_fn = tokenizer.tokenize
        rouge_eval = RougeEval(
            tokenizer_fn=tokenizer_fn
        )
        res_rouge = rouge_eval.eval(file_path=absolute_file_path)
        df["rouge1"].append(res_rouge["rouge1"]*100)
        df["rouge2"].append(res_rouge["rouge2"]*100)
        df["rougeL"].append(res_rouge["rougeL"]*100)
    
    # save result
    out_path = os.path.join(folder_path, f"results_{folder_name}.csv")
    pd.DataFrame(df).to_csv(out_path, index=False)

        