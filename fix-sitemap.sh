sed 's/http:\/\/localhost:4000/https:\/\/Gialbo\.github\.io/g' '_site/sitemap.xml' > tmp.xml
rm _site/sitemap.xml
cat tmp.xml
cp tmp.xml _site/sitemap.xml
rm tmp.xml