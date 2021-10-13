import pickle


def encode():
    secret = {
        "API_TOKEN": "YOUR API TOKEN",
        "BOT_NAME": "YOUR BOT NAME"
    }
    db = open("secret", "ab")
    pickle.dump(secret, db)
    db.close()


if __name__ == "__main__":
    encode()
