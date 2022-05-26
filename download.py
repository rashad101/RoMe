import gdown

def download_model():
    url = "https://drive.google.com/uc?id=1TXvzMQ6_BwMKFbTn8tviq-mZKqfKZF_y"
    output="saved_model/grammar.zip"
    gdown.download(url, output, quiet=False)

if __name__=="__main__":
    download_model()

