"""Abstract base class defining the loader interface."""

from abc import ABC, abstractmethod

from rag.loading.document import Document


class BaseLoader(ABC):
    """Interface every loader implements.

    Defining this explicitly means adding a new source format later (HTML,
    DOCX, a database) requires implementing one method, with no changes
    anywhere downstream.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents from the source.

        Returns:
            A list of Document objects.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"