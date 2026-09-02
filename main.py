from ptv_flow.cli import main


def _notify_finished() -> None:
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except Exception:
        print("\a", end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        _notify_finished()
