import glob


localSubPaths = ["test","train","valid"]
dir = "LicensePlateDatasetV2"

for subPath in localSubPaths:
    for txtFiles in glob.iglob(f'{dir}/{subPath}/labels/*'):
        if(not txtFiles.endswith(".txt")): continue
        with open(txtFiles, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False

        for line in lines:
            parts = line.strip().split()
            if(len(parts) == 0): continue
            if parts[0] != "0":
                parts[0] = "0"
                modified = False
            
            new_lines.append(" ".join(parts) + "\n")
        
        if(not modified): continue
        
        with open(txtFiles, "w") as f:
            f.writelines(new_lines)
