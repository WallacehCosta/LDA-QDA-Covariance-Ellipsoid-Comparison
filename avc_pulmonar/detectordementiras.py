def mentindo(txt):
    import random
    num = random.randint(1, 100)

    return f"\033[31:1mMentira detectada:\033[0m \033[97m{num}% de certeza sobre a frase\033[m '\033[97:1m{txt}\033[m'"

print(mentindo("Sou bilionário e namoro a Megan Fox"))
