import json
import os

def calculate_average_scores(concept_list, epoch,t,log_dir="./logs"):
    """
    计算指定日志文件中四个分数的平均值
    
    参数:
    concept_list: 概念列表
    log_dir: 日志文件所在的目录
    
    返回:
    包含四个分数平均值的字典
    """
    # 初始化四个分数的总和
    total_bleu_rea = 0
    total_ds_rea = 0
    total_bleu_dense_rea = 0
    total_ds_dense_rea = 0
    total_acc=0
    file_count = 0

    for concept in concept_list:
        path=os.path.join(log_dir,concept,f"epoch_{epoch}.json")
        # 检查目录是否存在
        if not os.path.exists(path):
            print(f"目录不存在: {path}")
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if t=='reasoning':
                    # 提取四个分数
                    bleu_rea = data.get('rea', {}).get('score', {}).get('bleu', 0)
                    ds_rea = data.get('rea', {}).get('score', {}).get('ds-score', 0)
                    bleu_dense_rea = data.get('dense-rea', {}).get('score', {}).get('bleu', 0)
                    ds_dense_rea = data.get('dense-rea', {}).get('score', {}).get('ds-score', 0)
                    
                    # 累加分数
                    total_bleu_rea += bleu_rea
                    total_ds_rea += ds_rea
                    total_bleu_dense_rea += bleu_dense_rea
                    total_ds_dense_rea += ds_dense_rea
                    file_count += 1
                    print(f"已处理文件: {path}")
                elif t=='base':

                    bleu_rea = data.get('vqa', {}).get('score', {}).get('bleu', 0)
                    ds_rea = data.get('vqa', {}).get('score', {}).get('ds-score', 0)
                    bleu_dense_rea = data.get('text_only', {}).get('score', {}).get('bleu', 0)
                    ds_dense_rea = data.get('text_only', {}).get('score', {}).get('ds-score', 0)
                    acc = data.get('rec', {}).get('score', {}).get('accuracy', 0)
                    
                    # 累加分数
                    total_bleu_rea += bleu_rea
                    total_ds_rea += ds_rea
                    total_bleu_dense_rea += bleu_dense_rea
                    total_ds_dense_rea += ds_dense_rea
                    total_acc+=acc
                    file_count += 1
                    print(f"已处理文件: {path}")
        except Exception as e:
            print(f"处理文件 {path} 时出错: {e}")
    
    if file_count == 0:
        print("没有找到有效的JSON文件")
        return None
    
    # 计算平均值
    avg_bleu_rea = total_bleu_rea / file_count
    avg_ds_rea = total_ds_rea / file_count
    avg_bleu_dense_rea = total_bleu_dense_rea / file_count
    avg_ds_dense_rea = total_ds_dense_rea / file_count
    avg_acc=total_acc / file_count
    
    return {
        "rea.score.bleu平均值": avg_bleu_rea,
        "rea.score.ds-score平均值": avg_ds_rea,
        "dense-rea.score.bleu平均值": avg_bleu_dense_rea,
        "dense-rea.score.ds-score平均值": avg_ds_dense_rea,
        "acc":avg_acc
    }

# 使用示例
if __name__ == "__main__":
    # 示例概念列表
    concept_list=['adrien_brody','b_jordan','butin','coco','dunpai','fine_woolfhard','gold_pineapple','leonardo','maeve_dog','mam','ningning','pig_cup','wangkai']
    # concept_list=['coco','dunpai']
    
    # 计算平均值
    averages = calculate_average_scores(concept_list,2,'reasoning',log_dir='./logs_reasoning')
    
    if averages:
        print("\n各分数的平均值:")
        for key, value in averages.items():
            print(f"{key}: {value:.4f}")
