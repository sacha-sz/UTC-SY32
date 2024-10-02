import os

def main(): 
    v = int(input("Quel est le numéro de la prédiciton :"))
    res_folder = os.path.join(os.getcwd(), 'predictions', f'roi_v{v}_test')

    fichiers = [f for f in os.listdir(res_folder) if f.endswith('.txt')]

    correspondance = {
        "none": 0,
        "frouge": 1,
        "fvert": 2,
        "forange": 3,
        "interdiction": 4,
        "danger": 5,
        "stop": 6,
        "ceder": 7,
        "obligation": 8
    }

    reversed_correspondance = {v: k for k, v in correspondance.items()}

    with open(f'export_roi_{v}.csv', 'w') as outfile:
        for fic in fichiers:
            with open(os.path.join(res_folder, fic), 'r') as f:
                for ligne in f:
                    ligne = ligne.strip()
                    if ligne:
                        item = ligne.split(' ')
                        one_line = fic[:-4] + ','
                        for i in range(len(item) - 1):
                            one_line += str(item[i]) + ','
                        one_line += str(reversed_correspondance[int(item[-1])]) + '\n'
                        outfile.write(one_line)
    print(f"Export successfull to : export_roi_{v}.csv")

if __name__ == "__main__":
    main()