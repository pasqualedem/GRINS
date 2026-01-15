import argparse

def main():
    parser = argparse.ArgumentParser(description="Run GRINS demo applications.")
    parser.add_argument(
        "--app",
        type=str,
        choices=["ranking", "annotate"],
        default="ranking",
        help="The demo application to run.",
    )
    args = parser.parse_args()

    demo(args.app)
    
def demo(app):
    """
    Run a Streamlit demo application.
    """
    print(f"Running the {app} demo application.")
    if app == "ranking":
        import grins.demo.ranking as ranking_app
        ranking_app.main()
    elif app == "annotate":
        import grins.demo.annotate as annotate_app
        annotate_app.main()
    else:
        raise ValueError(f"Unknown app: {app}")

if __name__ == "__main__":
    main()