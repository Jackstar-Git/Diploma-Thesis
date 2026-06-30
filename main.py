from gui.app import app
from modules.helpers import fetch_books_api, print_recommended_books
from sys import argv    


gui = True
args = argv[1:]
if "--no-gui" in args:
    gui = False
    


if __name__ == "__main__":
    if gui:
        app.run(debug=True, host="127.0.0.1", port=5000)
    else:
        preferences = (300, "J.K. Rowling", "Fantasy", "Magic, Adventure", "But his fortune changes when he receives a letter that tells him the truth about himself: he's a wizard. A mysterious visitor rescues him from his relatives and takes him to his new home, Hogwarts School of Witchcraft and Wizardry") #Test values
        results = fetch_books_api(preferences, 5)
        print_recommended_books(sorted_by="similarity", num_books=5, acented=False, num_places=3)
        
