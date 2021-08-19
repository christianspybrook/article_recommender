from flask import Flask, request
import pdf_converter as converter


app = Flask(__name__)

# home screen greeting
@app.route('/')
def home():

	return HOME_HTML

# home screen display
HOME_HTML = """
	<html><body>
		<h1>Welcome to the Arxiv scientific article recommender.</h1>
		<h2>Please enter an article number to find similar articles...</h2>
		<form action="/results">
			<input type='text' name='article_num'><br>
			<input type='submit' value='Get Recommendations'>
		 </form>
	</body></html>"""

# @app.route('/')
# def home():
# 	if request.args.get('article_num', ''):
# 		return HOME_HTML + WORKING_HTML
# 	else:
# 		return HOME_HTML

WORKING_HTML = """
	<html><body>
		<h2>Working...</h2>
	</body></html>"""

# results screen display
@app.route('/results')
def get_results():
	article_num = request.args.get('article_num', '')
	query_url = converter.get_url(article_num)
	converter.get_pdf(query_url)
	converter.call_converter()
	converter.clean_file()

	return RESULTS_HTML.format(query_url)

RESULTS_HTML = """
	<html><body>
		<h2>Your article was found at <a href={0}>{0}</a></h2>
		<h2>Five similar articles can be found at...</h2>
		<h2><a href={0}>{0}</a></h2>
		<h2><a href={0}>{0}</a></h2>
		<h2><a href={0}>{0}</a></h2>
		<h2><a href={0}>{0}</a></h2>
		<h2><a href={0}>{0}</a></h2>
	</body></html>
	"""

# http://192.168.0.156:5000/

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True , use_reloader=True)
