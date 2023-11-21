from models.Localizer import Localizer

if __name__ == '__main__':

    rna = 'ACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGACAGCUAUCAUCGA'

    rna = rna.replace('U', 'T')
    localizer = Localizer('finalResult/best.pt', 'finalResult/shapeInfo.pkl')
    score = localizer.predict(rna)
    print(f'the score is {score[0][0]:.2f}, localized in {"nuclues" if score >= 0.5 else "cytoplasm"}')
