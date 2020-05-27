import logging
logging.basicConfig(level=logging.INFO)
import subprocess  # Permite manipular directamente archivos de terminal


logger = logging.getLogger(__name__)
news_sites_uid = ['eluniversal', 'elpais']


def _extract():
    logging.info('Starting extract process')
    for news_site_uid in news_sites_uid:
        subprocess.run(['python', 'main.py', news_site_uid],
                       cwd='./extract')
        subprocess.run(['find', '.', '-name', f'{news_site_uid}*',
                        '-exec', 'mv', '{}', f'../transform/{news_site_uid}_.csv', ';'],
                       cwd='./extract')

def _transform():
    logging.info('Starting transform process')
    for news_site_uid in news_sites_uid:
        dirty_data_filename = f'{news_site_uid}_.csv'
        clean_data_filename = f'clean_{dirty_data_filename}'
        subprocess.run(['python', 'main.py', dirty_data_filename], 
                       cwd='./transform')
        subprocess.run(['rm', dirty_data_filename], cwd='./transform')
        subprocess.run(['mv', clean_data_filename, f'../load/{news_site_uid}.csv'],
                       cwd='./transform')


def _load():
    logger.info('Starting load process')
    for news_site_uid in news_sites_uid:
        clean_data_filename = f'{news_site_uid}.csv'
        subprocess.run(['python', 'main.py', clean_data_filename],
                      cwd='./load')
        subprocess.run(['rm', clean_data_filename], cwd='./load')


def main():
    _extract()
    _transform()
    _load()


if __name__ == '__main__':
    main()