import re

def test():
    tests = [
        '<parameter=path>./</parameter>',
        '<parameter="path">./</parameter>',
        '<parameter name="path">./</parameter>',
        '<parameter name=path>./</parameter>',
        '<parameter name=\'path\'>./</parameter>',
    ]
    pattern = r"<parameter(?:=|\s+name=)[\"']?([^>\"'\s]+)[\"']?[^>]*>(.*?)</parameter>"
    for t in tests:
        m = re.search(pattern, t, re.DOTALL)
        if m:
            print(f"MATCH: {m.group(1)} -> {m.group(2)}")
        else:
            print(f"FAIL: {t}")

test()
