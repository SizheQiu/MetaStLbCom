carve ../data/MAGs/lb_all/lb_nr.faa -u grampos --gapfill cheese --mediadb ../data/GSMMs/medium_db.tsv -o ../data/GSMMs/lb_raw.xml
echo "Recon for Lb done"
carve ../data/MAGs/st_all/st_nr.faa -u grampos --gapfill cheese --mediadb ../data/GSMMs/medium_db.tsv -o ../data/GSMMs/st_raw.xml
echo "Recon for St done"
