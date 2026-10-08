"""Streamlit AppTest with a random-weight model and synthetic gallery (no real data)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import CLASS_NAMES

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def app(monkeypatch, tiny_checkpoint, synthetic_gallery):
    monkeypatch.setenv("CUNEIFORM_CHECKPOINT", str(tiny_checkpoint))
    monkeypatch.setenv("CUNEIFORM_GALLERY_DIR", str(synthetic_gallery))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_renders_without_exception(app):
    assert not app.exception
    assert app.title[0].value.endswith("Cuneiform sign classifier")


def test_sidebar_lists_every_gallery_class(app):
    assert list(app.sidebar.selectbox[0].options) == CLASS_NAMES


def test_prediction_and_top3_are_shown(app):
    assert any("%" in m.value for m in app.markdown)
    assert len(app.sidebar.selectbox) == 2


def test_low_confidence_warning_is_shown_for_an_untrained_model(app):
    # 4 classes, random weights -> confidence close to 25 %, below the 50 % threshold
    assert len(app.warning) == 1
    assert "Low confidence" in app.warning[0].value


def test_changing_the_example_reruns_without_error(app):
    app.sidebar.selectbox[0].select("CC").run()
    assert not app.exception
