"""HPA parser."""

from bs4 import BeautifulSoup
import requests
from utils.logger import get_logger

logger = get_logger(__name__)


def get_image_urls(ens_id: str) -> dict[str, list[str]]:
    """Get nucleus, cytoplasm, protein URLs for an ENSG id"""
    logger.info("Getting the url of the images to download")
    HPA_url = f"https://v18.proteinatlas.org/{ens_id.split('.')[0]}.xml"
    try:
        response = requests.get(HPA_url, timeout=30)
        response.raise_for_status()
        xml_file = BeautifulSoup(response.text, "xml")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve XML for {ens_id}: {e}")
        return []

    im_urls = []
    for x in xml_file.find_all("cellExpression"):
        for y in x.find_all("data"):
            for n in y.find_all("cellLine"):
                for im in y.find_all("imageUrl"):
                    im_urls.append(
                        im.get_text()
                        .replace("_blue", "")
                        .replace("_red", "")
                        .replace("_green", "")
                        .replace("_blue_red_green", "")
                        .replace(".jpg", "")
                    )
    if not im_urls:
        logger.warning(f"No image URLs found for {ens_id}")
        return []
    urls_nuc = [x + "_blue.jpg" for x in im_urls]
    urls_cyt = [x + "_red.jpg" for x in im_urls]
    urls_prot = [x + "_green.jpg" for x in im_urls]
    urls_comb = [x + "_blue_red_green.jpg" for x in im_urls]
    logger.info(f"Found {len(im_urls)} image sets for {ens_id}")
    return list(zip(urls_nuc, urls_cyt, urls_prot, urls_comb))
