# INDICES = string.ascii_lowercase + string.ascii_uppercase
INDICES = ""
valid_characters = [(65, 65+26), (97, 97+26), (192, 214), (216, 246), (248, 328), (330, 383), (477, 687), (913, 974), (1024, 1119)]
for tup in valid_characters:
    for i in range(tup[0], tup[1]):
        INDICES += chr(i)