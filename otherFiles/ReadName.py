import pandas as pd

# 读取 CSV 文件
file_path = "../case_study_data/miRNA_name.csv"
df = pd.read_csv(file_path, header=None, names=["ID", "miRNA"])

# 将 ID 作为索引，创建字典
mirna_dict = dict(zip(df["ID"], df["miRNA"]))


def get_mirna_name(index):
    return mirna_dict[index]


        # diseasesIdx = (diseases[:8688] + 1).unsqueeze(1)
        # mirnasIdx = (mirnas[:8688] - self.num_diseases + 1).unsqueeze(1)
        #
        #
        # predict_score_first_half = predict_score[:8688]
        # predict_score_second_half = predict_score[8688:]
        #
        # predict_info_first_half = torch.cat((predict_score_first_half, mirnasIdx, diseasesIdx), dim=1)
        # predict_info_second_half = torch.cat((predict_score_second_half, mirnasIdx, diseasesIdx), dim=1)
        # predict_info = torch.cat((predict_info_first_half, predict_info_second_half), dim=0)
        #
        # # predict_info = predict_info.cpu()
        # # predict_info_df = pd.DataFrame(predict_info.detach().numpy())
        # # current_directory = os.getcwd()
        # # file_name = "predict_info.csv"
        # # file_path = os.path.join(current_directory, file_name)
        # # predict_info_df.to_csv(file_path, index=False)  # index=False 表示不保存索引
        # mirnasList = []
        # # for i in range(predict_info.size(0)):
        # #     mirna = get_mirna_name((predict_info[i][1]).int() + 1)
        # #     mirnasList.append(mirna)
        #
        # file_path = "case_study_data/miRNA_name.csv"
        # df = pd.read_csv(file_path, header=None, names=["ID", "miRNA"])
        # mirna_dict = dict(zip(df["ID"], df["miRNA"]))
        #
        # for i in range(predict_info.size(0)):
        #     mirnaName = mirna_dict[int(predict_info[i][1].item())]
        #     mirnasList.append(mirnaName)