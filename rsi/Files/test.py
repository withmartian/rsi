import argparse

def foo(flag=False):
    # function code here
    print(f"The value of flag is: {flag}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=bool, default=None)
    args = parser.parse_args()

    flag_value = args.flag if args.flag is not None else foo.__defaults__[0]
    foo(flag_value)