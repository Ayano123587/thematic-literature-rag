"""Root training entrypoint; normal Python source modules own implementation."""
import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["full"]:
        sys.argv.pop(1)
        from src.full_generator_training import main
    elif sys.argv[1:2] == ["reranker"]:
        sys.argv.pop(1)
        from src.reranker_training import main
    else:
        from src.narrative_training import main
    main()
