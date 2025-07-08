import subprocess

def main():
    print("=== Biometric Authentication System ===")
    print("1. Face Login")
    print("2. Fingerprint Login")
    choice = input("Select method (1 or 2): ")

    if choice == "1":
        print("\n[Using Face Recognition]")
        subprocess.run(["python", "login.py"])
    elif choice == "2":
        print("\n[Using Fingerprint Matching]")
        subprocess.run(["python", "fingerprint_auth.py"])
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
