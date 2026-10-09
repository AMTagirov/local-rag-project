import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import RAGConfig
from src.services.document_parser import create_document_parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argument_parser = argparse.ArgumentParser(
        description="Предварительный просмотр результата PDF-парсинга"
    )
    argument_parser.add_argument("document", help="Путь к PDF или DOCX")
    argument_parser.add_argument("--config", default="src/config/config.yaml")
    argument_parser.add_argument("--output", default="document_analysis_preview.md")
    args = argument_parser.parse_args()

    document_path = Path(args.document)
    if not document_path.exists():
        argument_parser.error(f"Документ не найден: {document_path}")
    if document_path.suffix.lower() != ".pdf":
        argument_parser.error("Для document analysis поддерживается только PDF")

    config = RAGConfig.from_yaml(args.config)
    parser = create_document_parser(config)
    pages = []

    for page_number, page_text in enumerate(parser.parse(str(document_path)), start=1):
        pages.append(f"# Результат {page_number}\n\n{page_text}")
        print(f"Страница/блок {page_number}: {len(page_text)} символов")

    if not pages:
        raise RuntimeError("Парсер не вернул содержимое")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n---\n\n".join(pages), encoding="utf-8")
    print(f"Результат сохранён: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
