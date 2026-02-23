"""Tests for literature.patent_search tool."""

import pytest
from unittest.mock import patch, MagicMock


class TestPatentSearchLens:
    """Test Lens.org API path (mocked)."""

    @patch("httpx.post")
    def test_lens_success(self, mock_post):
        from ct.tools.literature import patent_search

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "total": 42,
            "data": [
                {
                    "lens_id": "001-234-567-890",
                    "title": [{"text": "CRBN Molecular Glue Compounds"}],
                    "abstract": [{"text": "Novel molecular glue degraders targeting cereblon..."}],
                    "applicant": [{"name": "Pharma Corp"}],
                    "publication_date": "2023-06-15",
                    "doc_number": "US20230189102",
                    "jurisdiction": "US",
                    "kind": "A1",
                },
                {
                    "lens_id": "001-234-567-891",
                    "title": [{"text": "Bifunctional Degrader Molecules"}],
                    "abstract": [{"text": "Heterobifunctional small molecules..."}],
                    "applicant": [{"name": "BioTech Inc"}, {"name": "University X"}],
                    "publication_date": "2022-11-03",
                    "doc_number": "WO2022234567",
                    "jurisdiction": "WO",
                    "kind": "A1",
                },
            ],
        }
        mock_post.return_value = resp

        session = MagicMock()
        session.config.get.side_effect = lambda key, default=None: (
            "fake-lens-key" if key == "api.lens_key" else default
        )

        result = patent_search(query="CRBN molecular glue", max_results=20, _session=session)

        assert result["source"] == "lens.org"
        assert result["total_count"] == 42
        assert len(result["patents"]) == 2
        assert result["patents"][0]["title"] == "CRBN Molecular Glue Compounds"
        assert "Pharma Corp" in result["patents"][0]["applicants"]
        assert "summary" in result

    @patch("httpx.post")
    def test_lens_failure_falls_through(self, mock_post):
        """If Lens.org fails, should fall through to EPO or PubMed."""
        from ct.tools.literature import patent_search

        mock_post.return_value = MagicMock(status_code=401)

        session = MagicMock()
        session.config.get.side_effect = lambda key, default=None: (
            "bad-key" if key == "api.lens_key" else default
        )

        # This will try Lens (fail), then EPO (may fail in test), then PubMed fallback
        # We need to mock the downstream calls too
        with patch("httpx.get") as mock_get:
            # EPO also fails
            epo_resp = MagicMock()
            epo_resp.status_code = 403
            # PubMed search + summary
            search_resp = MagicMock()
            search_resp.json.return_value = {"esearchresult": {"count": "0", "idlist": []}}
            search_resp.raise_for_status = MagicMock()

            mock_get.side_effect = [epo_resp, search_resp]

            result = patent_search(query="test compound", _session=session)

            assert result["source"] == "pubmed_fallback"


class TestPatentSearchEPO:
    """Test EPO OPS path (mocked)."""

    @patch("httpx.get")
    def test_epo_success(self, mock_get):
        from ct.tools.literature import patent_search

        epo_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <ops:world-patent-data xmlns:ops="http://ops.epo.org"
                               xmlns:exch="http://www.epo.org/exchange">
          <ops:biblio-search total-result-count="5">
            <ops:search-result>
              <ops:publication-reference>
                <exch:exchange-documents>
                  <exch:exchange-document doc-number="20230189102" country="US" kind="A1">
                    <exch:bibliographic-data>
                      <exch:publication-reference>
                        <exch:document-id>
                          <exch:date>20230615</exch:date>
                        </exch:document-id>
                      </exch:publication-reference>
                      <exch:invention-title lang="en">Molecular Glue Degraders</exch:invention-title>
                      <exch:parties>
                        <exch:applicants>
                          <exch:applicant>
                            <exch:applicant-name>
                              <exch:name>PHARMA CORP</exch:name>
                            </exch:applicant-name>
                          </exch:applicant>
                        </exch:applicants>
                      </exch:parties>
                      <exch:abstract lang="en">
                        <exch:p>Novel molecular glue degrader compounds.</exch:p>
                      </exch:abstract>
                    </exch:bibliographic-data>
                  </exch:exchange-document>
                </exch:exchange-documents>
              </ops:publication-reference>
            </ops:search-result>
          </ops:biblio-search>
        </ops:world-patent-data>"""

        resp = MagicMock()
        resp.status_code = 200
        resp.text = epo_xml
        mock_get.return_value = resp

        result = patent_search(query="molecular glue degrader")

        assert result["source"] == "epo_ops"
        assert result["total_count"] == 5
        assert len(result["patents"]) >= 1
        assert result["patents"][0]["title"] == "Molecular Glue Degraders"
        assert "PHARMA CORP" in result["patents"][0]["applicants"]
        assert "summary" in result

    @patch("httpx.get")
    def test_epo_no_results(self, mock_get):
        """EPO returning 404 should fall through to PubMed."""
        from ct.tools.literature import patent_search

        # EPO 404
        epo_resp = MagicMock()
        epo_resp.status_code = 404

        # PubMed search
        pm_search_resp = MagicMock()
        pm_search_resp.json.return_value = {
            "esearchresult": {"count": "2", "idlist": ["111", "222"]},
        }
        pm_search_resp.raise_for_status = MagicMock()

        # PubMed summary
        pm_summary_resp = MagicMock()
        pm_summary_resp.json.return_value = {
            "result": {
                "uids": ["111", "222"],
                "111": {
                    "title": "Patent landscape review",
                    "authors": [{"name": "Doe J"}],
                    "source": "J Med Chem",
                    "pubdate": "2024",
                    "articleids": [],
                },
                "222": {
                    "title": "IP analysis of degraders",
                    "authors": [{"name": "Smith A"}],
                    "source": "Drug Discov Today",
                    "pubdate": "2023",
                    "articleids": [],
                },
            },
        }
        pm_summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [epo_resp, pm_search_resp, pm_summary_resp]

        result = patent_search(query="xyznonexistent")

        assert result["source"] == "pubmed_fallback"
        assert result["total_count"] == 2

    @patch("httpx.get")
    def test_epo_rate_limited(self, mock_get):
        """EPO 403 should fall through to PubMed."""
        from ct.tools.literature import patent_search

        epo_resp = MagicMock()
        epo_resp.status_code = 403

        pm_resp = MagicMock()
        pm_resp.json.return_value = {"esearchresult": {"count": "0", "idlist": []}}
        pm_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [epo_resp, pm_resp]

        result = patent_search(query="test")

        assert result["source"] == "pubmed_fallback"


class TestPatentSearchPubMedFallback:
    """Test PubMed fallback path."""

    @patch("httpx.get")
    def test_pubmed_fallback_adds_patent_terms(self, mock_get):
        """PubMed fallback should add patent-related terms to query."""
        from ct.tools.literature import patent_search

        # EPO fails
        epo_resp = MagicMock()
        epo_resp.status_code = 500

        # PubMed search
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "esearchresult": {"count": "3", "idlist": ["100"]},
        }
        search_resp.raise_for_status = MagicMock()

        # PubMed summary
        summary_resp = MagicMock()
        summary_resp.json.return_value = {
            "result": {
                "uids": ["100"],
                "100": {
                    "title": "Patent review: CRBN degraders",
                    "authors": [{"name": "Jones B"}],
                    "source": "Expert Opin Ther Pat",
                    "pubdate": "2024",
                    "articleids": [{"idtype": "doi", "value": "10.1234/test"}],
                },
            },
        }
        summary_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [epo_resp, search_resp, summary_resp]

        result = patent_search(query="CRBN degrader")

        assert result["source"] == "pubmed_fallback"
        assert "note" in result  # Should include note about fallback
        assert len(result["articles"]) == 1
        assert "summary" in result

    @patch("httpx.get")
    def test_pubmed_fallback_no_results(self, mock_get):
        """PubMed fallback with no results."""
        from ct.tools.literature import patent_search

        epo_resp = MagicMock()
        epo_resp.status_code = 500

        pm_resp = MagicMock()
        pm_resp.json.return_value = {"esearchresult": {"count": "0", "idlist": []}}
        pm_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [epo_resp, pm_resp]

        result = patent_search(query="xyznonexistent_patent")

        assert result["source"] == "pubmed_fallback"
        assert result["total_count"] == 0


class TestPatentSearchNoSession:
    """Test patent_search without a session (no API keys)."""

    @patch("httpx.get")
    def test_no_session_skips_lens(self, mock_get):
        """Without session, should skip Lens.org and try EPO directly."""
        from ct.tools.literature import patent_search

        epo_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <ops:world-patent-data xmlns:ops="http://ops.epo.org"
                               xmlns:exch="http://www.epo.org/exchange">
          <ops:biblio-search total-result-count="1">
            <ops:search-result>
              <ops:publication-reference>
                <exch:exchange-documents>
                  <exch:exchange-document doc-number="123" country="EP" kind="A1">
                    <exch:bibliographic-data>
                      <exch:invention-title lang="en">Test Patent</exch:invention-title>
                    </exch:bibliographic-data>
                  </exch:exchange-document>
                </exch:exchange-documents>
              </ops:publication-reference>
            </ops:search-result>
          </ops:biblio-search>
        </ops:world-patent-data>"""

        resp = MagicMock()
        resp.status_code = 200
        resp.text = epo_xml
        mock_get.return_value = resp

        result = patent_search(query="kinase inhibitor")

        assert result["source"] == "epo_ops"
        assert "summary" in result
