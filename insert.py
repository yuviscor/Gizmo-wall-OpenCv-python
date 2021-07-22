import sqlite3



def add(facepic):

    connection = sqlite3.connect("face.db",check_same_thread=False)
    cursor = connection.cursor()
    print("photo added in the database ")

    cursor.execute('INSERT INTO FACE VALUES(?)', (facepic,))
    connection.commit()
    cursor.close()
    connection.close()

