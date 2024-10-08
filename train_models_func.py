def train_models_a(self):
    """Trains all models and evaluates them."""
    for name, model in self.models.items():
        print(f"Training {name}...")
        clf = Pipeline(steps=[('model', model)])
        clf.fit(self.X_train, self.y_train)
        y_pred_train = clf.predict(self.X_train)
        y_pred_test = clf.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)

        self.model_results[name] = {
            'model': clf,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_pred_test': y_pred_test,
        }

        if self.display_analytics:
            print(f"\n{name} Results:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred_test))
            print("Confusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred_test)
            print(cm)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, cbar_kws={'label': 'Scale'}, vmin=0, vmax=705)
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.show()

        # Store trained model
        self.trained_models[name] = clf

def train_models_b(self):
    """Trains all models, evaluates them, and displays analytics in subplots."""
    
    # Create subplots (4 rows, 2 columns) for confusion matrices
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten the grid of axes into a 1D array to iterate over
    
    for idx, (name, model) in enumerate(self.models.items()):
        print(f"Training {name}...")
        clf = Pipeline(steps=[('model', model)])
        clf.fit(self.X_train, self.y_train)
        y_pred_train = clf.predict(self.X_train)
        y_pred_test = clf.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)

        self.model_results[name] = {
            'model': clf,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_pred_test': y_pred_test,
        }

        if self.display_analytics:
            print(f"\n{name} Results:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred_test))

            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred_test)
            print("Confusion Matrix:")
            print(cm)
            
            # Plotting confusion matrix in a subplot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, 
                        cbar_kws={'label': 'Scale'}, vmin=0, vmax=705, ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')

        # Store trained model
        self.trained_models[name] = clf
    
    # Remove unused subplots (if there are fewer than 8 models)
    if len(self.models) < len(axes):
        for i in range(len(self.models), len(axes)):
            fig.delaxes(axes[i])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def train_models_c(self):
    """Trains all models, evaluates them, and displays analytics in subplots with two matrices per row."""

    num_models = len(self.models)
    rows = (num_models + 1) // 2  # Calculate the number of rows needed for 2 matrices per row

    # Create a GridSpec layout: 2 rows per model, and 2 models per row (columns)
    fig = plt.figure(figsize=(16, rows * 6))  # Adjust height dynamically based on number of rows
    gs = GridSpec(2 * rows, 2, figure=fig)  # 2 rows per model, 2 models per row

    for idx, (name, model) in enumerate(self.models.items()):
        print(f"Training {name}...")
        clf = Pipeline(steps=[('model', model)])
        clf.fit(self.X_train, self.y_train)
        y_pred_train = clf.predict(self.X_train)
        y_pred_test = clf.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        clf_report = classification_report(self.y_test, y_pred_test, output_dict=True)

        self.model_results[name] = {
            'model': clf,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_pred_test': y_pred_test,
        }

        # Calculate the position in GridSpec
        row_idx = (idx // 2) * 2  # Two rows for each set of matrices (matrix + text)
        col_idx = idx % 2  # Switch between 0 (left column) and 1 (right column)

        # GridSpec setup: Confusion matrix in the top half
        ax_cm = fig.add_subplot(gs[row_idx, col_idx]);  # Top row of each pair for confusion matrix

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, 
                    cbar_kws={'label': 'Scale'}, vmin=0, vmax=705, ax=ax_cm)
        ax_cm.set_title(f'{name} Confusion Matrix')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')

        # Text section in the bottom half
        ax_text = fig.add_subplot(gs[row_idx + 1, col_idx])  # Bottom row of each pair for text

        # Add training accuracy, test accuracy, and a brief classification report in compact format
        text_info = (f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}\n"
                     f"Precision: {clf_report['weighted avg']['precision']:.4f} | "
                     f"Recall: {clf_report['weighted avg']['recall']:.4f} | "
                     f"F1-Score: {clf_report['weighted avg']['f1-score']:.4f}")

        ax_text.text(0.5, 0.5, text_info, ha='center', va='center', fontsize=12)
        ax_text.axis('off')  # Hide the axes for the text subplot

        # Store trained model
        self.trained_models[name] = clf

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def train_models_d(self):
    """Trains all models, evaluates them, and displays analytics in subplots with borders and better layout."""

    num_models = len(self.models)
    rows = (num_models + 1) // 2  # Calculate the number of rows needed for 2 matrices per row

    # Create a GridSpec layout: 2 rows per model, and 2 models per row (columns)
    fig = plt.figure(figsize=(16, rows * 6))  # Adjust height dynamically based on number of rows
    gs = GridSpec(2 * rows, 2, figure=fig)  # 2 rows per model, 2 models per row

    # Ensure background contrast (white background)
    fig.patch.set_facecolor('white')  # Set figure background color

    for idx, (name, model) in enumerate(self.models.items()):
        print(f"Training {name}...")
        clf = Pipeline(steps=[('model', model)])
        clf.fit(self.X_train, self.y_train)
        y_pred_train = clf.predict(self.X_train)
        y_pred_test = clf.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        clf_report = classification_report(self.y_test, y_pred_test, output_dict=True)

        self.model_results[name] = {
            'model': clf,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_pred_test': y_pred_test,
        }

        # Calculate the position in GridSpec
        row_idx = (idx // 2) * 2  # Two rows for each set of matrices (matrix + text)
        col_idx = idx % 2  # Switch between 0 (left column) and 1 (right column)

        # GridSpec setup: Confusion matrix in the top half
        ax_cm = fig.add_subplot(gs[row_idx, col_idx])  # Top row of each pair for confusion matrix

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, 
                    cbar_kws={'label': 'Scale'}, vmin=0, vmax=705, ax=ax_cm)
        ax_cm.set_title(f'{name} Confusion Matrix')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')

        # Draw grid lines manually around the subplot to make them more visible
        for side in ['top', 'bottom', 'left', 'right']:
            ax_cm.spines[side].set_linewidth(2)
            ax_cm.spines[side].set_color('black')

        # Text section in the bottom half
        ax_text = fig.add_subplot(gs[row_idx + 1, col_idx])  # Bottom row of each pair for text

        # Add training accuracy, test accuracy, and a brief classification report in compact format
        text_info = (f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}\n"
                     f"Precision: {clf_report['weighted avg']['precision']:.4f} | "
                     f"Recall: {clf_report['weighted avg']['recall']:.4f} | "
                     f"F1-Score: {clf_report['weighted avg']['f1-score']:.4f}")

        ax_text.text(0.5, 0.5, text_info, ha='center', va='center', fontsize=12)
        ax_text.axis('off')  # Hide the axes for the text subplot

        # Draw manual lines to outline the text box more clearly
        for side in ['top', 'bottom', 'left', 'right']:
            ax_text.spines[side].set_linewidth(2)
            ax_text.spines[side].set_color('black')

        # Store trained model
        self.trained_models[name] = clf

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()