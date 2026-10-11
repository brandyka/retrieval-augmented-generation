from langchain_community.document_loaders import PyPDFLoader

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_pdf(self):
        loader = PyPDFLoader(
            file_path=self.file_path,
            extraction_mode="layout"

        )
        documents = loader.load()
        return documents

