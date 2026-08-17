from cli import predict_single
from gui import run_gui

def main():
    while True:
        print("\nWelcome to Diabetes Prediction AI")
        print("Select an option:")
        print("1: CLI (Command-Line Interface)")
        print("2: GUI (Graphical User Interface)")
        print("3: Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == '3':
            print("Exiting program. Goodbye!")
            break
        elif choice not in ['1', '2']:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        if choice == '1':
            print("\nCLI Mode Selected")
            predict_single()
        elif choice == '2':
            print("\nGUI Mode Selected")
            run_gui()

if __name__ == "__main__":
    main()