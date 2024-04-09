from tkinter import *


def get_contents():
    try:
        with open('cover_letter.txt', mode='r') as file:
            contents = file.read()
    except FileNotFoundError:
        with open('cover_letter.txt', mode='w') as file:
            contents = ''
            file.write(contents)
    finally:
        return contents


def write_letter():
    with open('cover_letter.txt', mode='r') as file:
        contents = file.read()

    contents = contents.replace('[position]', position.get().title())
    contents = contents.replace('[profession]', profession.get())
    contents = contents.replace('[company name]', company.get().title())

    with open(f'{company.get().title()}_cover_letter.txt', mode='w') as file:
        file.write(contents)

    position.delete(0, END)
    profession.delete(0, END)
    company.delete(0, END)

window = Tk()
window.minsize(width=200, height=200)
window.config(padx=20, pady=20)
window.title('Cover Letter')

letter = get_contents()

pos_label = Label(text="Position")
pos_label.grid(row=0, column=0)
comp_label = Label(text='Company')
comp_label.grid(row=1, column=0)
prof_label = Label(text="Profession")
prof_label.grid(row=2, column=0)

position = Entry()
position.grid(row=0, column=1)
company = Entry()
company.grid(row=1, column=1)
profession = Entry()
profession.grid(row=2, column=1)

button = Button(text="Write Letter", command=write_letter)
button.grid(row=3, column=1)

window.mainloop()
