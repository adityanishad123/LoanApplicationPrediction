/* ==========================================================
   LoanWise — script.js
   Client-side validation + submit-loading UX
   ========================================================== */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loanForm");
  const submitBtn = document.getElementById("submitBtn");
  const submitText = document.getElementById("submitText");
  const spinner = document.getElementById("spinner");
  const resetBtn = document.getElementById("resetBtn");

  if (!form) return;

  const rules = {
    age: { min: 18, max: 100, type: "number", label: "Age" },
    income: { min: 1, type: "number", label: "Annual Income" },
    dependents: { min: 0, max: 10, type: "number", label: "Number of Dependents" },
    loan_amount: { min: 1, type: "number", label: "Loan Amount" },
    credit_score: { min: 300, max: 900, type: "number", label: "Credit Score" },
  };

  function setError(name, message) {
    const span = form.querySelector(`.error-text[data-for="${name}"]`);
    const field = form.querySelector(`[name="${name}"]`);
    if (span) span.textContent = message || "";
    if (field && field.classList) {
      field.classList.toggle("invalid", Boolean(message));
    }
  }

  function validateField(name) {
    const field = form.querySelector(`[name="${name}"]`);
    if (!field) return true;

    // Radio groups
    const radios = form.querySelectorAll(`input[name="${name}"][type="radio"]`);
    if (radios.length) {
      const checked = Array.from(radios).some((r) => r.checked);
      setError(name, checked ? "" : "Please select an option.");
      return checked;
    }

    const value = field.value.trim();
    if (value === "") {
      setError(name, "This field is required.");
      return false;
    }

    const rule = rules[name];
    if (rule && rule.type === "number") {
      const num = Number(value);
      if (Number.isNaN(num)) {
        setError(name, `${rule.label} must be a number.`);
        return false;
      }
      if (rule.min !== undefined && num < rule.min) {
        setError(name, `${rule.label} must be at least ${rule.min}.`);
        return false;
      }
      if (rule.max !== undefined && num > rule.max) {
        setError(name, `${rule.label} must be at most ${rule.max}.`);
        return false;
      }
    }

    setError(name, "");
    return true;
  }

  const fieldNames = [
    "age", "gender", "income", "employment", "education", "marital",
    "dependents", "loan_amount", "loan_term", "credit_score",
    "existing_loan", "property_area",
  ];

  // Live validation as the user types / selects
  fieldNames.forEach((name) => {
    const els = form.querySelectorAll(`[name="${name}"]`);
    els.forEach((el) => {
      el.addEventListener("input", () => validateField(name));
      el.addEventListener("change", () => validateField(name));
    });
  });

  form.addEventListener("submit", (e) => {
    let isValid = true;
    fieldNames.forEach((name) => {
      const ok = validateField(name);
      if (!ok) isValid = false;
    });

    if (!isValid) {
      e.preventDefault();
      const firstError = form.querySelector(".error-text:not(:empty)");
      if (firstError) firstError.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    // Show loading state while the server processes the prediction
    submitBtn.disabled = true;
    submitText.textContent = "Predicting...";
    spinner.classList.remove("hidden");
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      fieldNames.forEach((name) => setError(name, ""));
    });
  }
});
