import requests


def download_rawtext(url: str, outfile: str = None) -> str:
    """Download raw text from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
    else:
        print(f"Failed to download {url}. HTTP Status: {response.status_code}")
        text = None

    if text and outfile:
        with open(outfile, "w") as f:
            f.write(text)
        print(f"File downloaded: {outfile}")
    return text


#####ANCHOR Fetch JS file
def main():
    download_rawtext(
        url="https://raw.githubusercontent.com/thangckt/stuff/refs/heads/js_iplog/gas_ip_logger.js",
        outfile="_docs/1thang_js/gas_ip_logger.js",
    )


if __name__ == "__main__":
    main()
