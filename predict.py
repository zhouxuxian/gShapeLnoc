from models.Localizer import Localizer

if __name__ == '__main__':
    #预测示例代码
    rna = 'ACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGA'

    rna = rna.replace('U', 'T')
    #加载训练好的权重
    localizer = Localizer('finalResult/best.pt', 'finalResult/shapeInfo.pkl')
    #开始预测
    score = localizer.predict(rna)
    print(f'the score is {score[0][0]:.2f}, localized in {"nuclues" if score >= 0.5 else "cytoplasm"}')
